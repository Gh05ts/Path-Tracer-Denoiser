import pytest

torch = pytest.importorskip("torch")

from wavefront_denoiser.temporal import reproject_history,reproject_history_gather


@pytest.mark.parametrize("shape",[(1,1),(1,5),(5,1),(5,7)])
def test_gather_warp_matches_bilinear_values_and_masks(shape):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h,w = shape
    image = torch.rand(2,3,h,w,device=device,dtype=torch.double)
    motion = torch.full((2,2,h,w),.17,device=device,dtype=torch.double)
    mask = torch.ones(2,1,h,w,device=device,dtype=torch.double)
    a,va = reproject_history(image,motion,mask)
    b,vb = reproject_history_gather(image,motion,mask)
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert torch.equal(va,vb)


def test_gather_warp_gradcheck_under_deterministic_algorithms():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    before = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        image = torch.rand(1,2,3,4,device=device,dtype=torch.double,requires_grad=True)
        motion = torch.full((1,2,3,4),.17,device=device,dtype=torch.double,requires_grad=True)
        mask = torch.ones(1,1,3,4,device=device,dtype=torch.double)
        assert torch.autograd.gradcheck(lambda a,b:reproject_history_gather(a,b,mask)[0],(image,motion))
    finally:
        torch.use_deterministic_algorithms(before)
