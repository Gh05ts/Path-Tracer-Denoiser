"""A shared R2 coordinator: one durable claim per whole clip."""

from pathlib import Path
import time
import uuid

from .plan import prefix, identifier


class Coordinator:
    def __init__(self, store, plan):
        if plan.get("format") != "wavefront-distributed-clips" or plan.get("version") != 1:
            raise ValueError("Unsupported distributed capture plan")
        self.store, self.plan = store, plan
        self.prefix = prefix(plan["run_id"])

    def key(self, group):
        if group not in self.plan["tasks"]:
            raise ValueError("Unknown distributed clip")
        return f"{self.prefix}/tasks/{identifier(group)}.json"

    def initialize(self, prepared):
        prepared = Path(prepared)
        binding = {"capture_lock_sha256": self.plan["capture_lock_sha256"], "run_id": self.plan["run_id"]}
        if not self.store.put("_distributed/active.json", binding):
            raise ValueError("Another distributed queue already manages this dataset prefix")
        if not self.store.put(f"{self.prefix}/plan.json", self.plan):
            raise ValueError("An incompatible distributed plan already uses this run ID")
        ready = self.store.get(f"{self.prefix}/ready.json")
        if ready is not None:
            if ready[0] != binding:
                raise ValueError("Distributed coordinator binding changed")
            return
        remote = self.store.client.list_files()
        for scene, receipt in self.plan["archived_scenes"].items():
            existing = self.store.get(f"r2-archives/{scene}.json")
            if existing is None or existing[0] != receipt:
                raise ValueError(f"Existing remote scene receipt disagrees with the local dataset: {scene}")
        for group in self.plan["task_order"]:
            task = self.plan["tasks"][group]
            seed = task["seed"]
            if seed:
                if any(not self.store.client.matches(remote.get(n), h) for n, h in seed["files"].items()):
                    raise ValueError(f"Cannot seed an unverified archived clip: {group}")
            state = {"group": group, "capture_lock_sha256": self.plan["capture_lock_sha256"],
                     **(seed or {"status": "pending"})}
            current = self.store.get(self.key(group))
            if current is None:
                if not self.store.put(self.key(group), state):
                    raise ValueError("Concurrent initialization changed a task")
            elif current[0].get("capture_lock_sha256") != self.plan["capture_lock_sha256"]:
                raise ValueError("Existing task belongs to another dataset")
        self.store.client.upload_files(prepared / "metadata", list(self.plan["metadata_files"]), self.plan["metadata_files"])
        self.store.client.verify_files(self.plan["metadata_files"])
        self.store.upload(prepared / "metadata.tar", f"{self.prefix}/metadata.tar", self.plan["bundle"])
        if not self.store.put(f"{self.prefix}/ready.json", binding):
            raise ValueError("Distributed coordinator initialization conflict")

    def require_ready(self):
        active = self.store.get("_distributed/active.json")
        if active is None or active[0] != {"run_id": self.plan["run_id"],
                                          "capture_lock_sha256": self.plan["capture_lock_sha256"]}:
            raise ValueError("A different distributed queue manages this dataset prefix")
        record = self.store.get(f"{self.prefix}/ready.json")
        if record is None or record[0].get("capture_lock_sha256") != self.plan["capture_lock_sha256"]:
            raise ValueError("Coordinator initialization has not completed")

    def claim(self, worker, machine, *, exclude=()):
        identifier(worker)
        self.require_ready()
        states = []
        for group in self.plan["task_order"]:
            task = self.plan["tasks"][group]
            if group in exclude:
                continue
            record = self.store.get(self.key(group))
            if record is None:
                raise ValueError("Coordinator task is missing")
            state, etag = record
            if state.get("capture_lock_sha256") != self.plan["capture_lock_sha256"]:
                raise ValueError("Coordinator task provenance changed")
            if state["status"] == "claimed" and state["worker"] == worker:
                if state["machine"] != machine:
                    raise ValueError("Worker ID is already owned by a different local workspace")
                return state
            if state["status"] == "pending" and task["affinity"] in (None, worker):
                states.append((group, state, etag))
        for group, state, etag in states:
            claimed = dict(state, status="claimed", worker=worker, machine=machine,
                           token=uuid.uuid4().hex, phase="rendering", updated_unix=time.time())
            if self.store.put(self.key(group), claimed, etag=etag):
                return claimed
        return None

    def owned(self, claim):
        record = self.store.get(self.key(claim["group"]))
        if record is None or record[0].get("status") != "claimed" or record[0].get("token") != claim["token"]:
            raise ValueError("Clip ownership was lost; refusing to publish results")
        return record

    def phase(self, claim, phase):
        state, etag = self.owned(claim)
        updated = dict(state, phase=phase, updated_unix=time.time())
        if not self.store.put(self.key(claim["group"]), updated, etag=etag):
            raise ValueError("Clip ownership changed during phase update")

    def complete(self, claim, inventory):
        group = claim["group"]
        task = self.plan["tasks"][group]
        required = set()
        for index in range(8):
            frame = f"{group}/frame_{index:06d}"
            required.update(f"{frame}/{name}" for name in ("frame.exr", "manifest.json"))
            if task["layers"]:
                required.update(f"sample_layers/{frame}/{name}" for name in ("layers.exr", "manifest.json"))
            if task["decoded"]:
                required.update(f"decoded/{frame}/{name}" for name in ("float.npy", "uint.npy", "manifest.json"))
        allowed = required | {f"logs/{group}.log", f"logs/layers-{group}.log"}
        if required - inventory.keys() or inventory.keys() - allowed:
            raise ValueError("Completion must include the whole clip and only its required files/logs")
        state, etag = self.owned(claim)
        self.store.verify(inventory)
        complete = dict(state, status="complete", phase="verified", files=inventory)
        if not self.store.put(self.key(claim["group"]), complete, etag=etag):
            raise ValueError("Clip ownership changed before completion")
        readback = self.store.get(self.key(claim["group"]))
        if readback is None or readback[0] != complete:
            raise ValueError("Clip completion could not be verified; retained local files")

    def finalize_scene(self, scene, *, destination, endpoint):
        # Existing BMW receipts keep their original integrator/revision binding.
        if scene in self.plan["archived_scenes"]:
            return False
        inventory = dict(self.plan["metadata_files"])
        for group, task in self.plan["tasks"].items():
            if task["scene"] != scene:
                continue
            record = self.store.get(self.key(group))
            if record is None or record[0]["status"] != "complete":
                return False
            for name, hashes in record[0]["files"].items():
                if name in inventory and inventory[name] != hashes:
                    raise ValueError("Conflicting distributed scene inventory")
                inventory[name] = hashes
        self.store.client.verify_files(inventory)
        receipt = {"format": "wavefront-r2-scene-archive", "version": 1, "status": "verified",
                   "scene": scene, "destination": destination, "r2_endpoint": endpoint,
                   "capture_lock_sha256": self.plan["capture_lock_sha256"], "files": inventory,
                   "distributed_run_id": self.plan["run_id"]}
        if not self.store.put(f"r2-archives/{identifier(scene)}.json", receipt):
            raise ValueError("A different scene receipt already exists; retained distributed results")
        return True

    def release(self, group, token):
        """Explicit recovery only, after stopping the old worker; no lease theft."""
        record = self.store.get(self.key(group))
        if record is None or record[0]["status"] != "claimed" or record[0]["token"] != token:
            raise ValueError("Release requires the current claim token")
        pending = {"group": group, "capture_lock_sha256": self.plan["capture_lock_sha256"], "status": "pending"}
        if not self.store.put(self.key(group), pending, etag=record[1]):
            raise ValueError("Claim changed before release")

    def status(self):
        states = {group: self.store.get(self.key(group))[0] for group in self.plan["tasks"]}
        return {"counts": {name: sum(s["status"] == name for s in states.values())
                           for name in ("pending", "claimed", "complete")},
                "claimed": [{k: s[k] for k in ("group", "worker", "phase", "token", "updated_unix")}
                            for s in states.values() if s["status"] == "claimed"]}
