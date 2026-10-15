"""Server tests: OpenEB histogram bin boundaries, spatial counts, and contracts."""
import torch
import pytest
from event_state.data.event_representation import EventVoxelizer
from event_state.data.transforms import PairedSequenceTransform
from event_state.data.dsec import event_representation_metadata


def test_histogram_boundaries_polarity_and_empty():
    voxel = EventVoxelizer(height=1,width=1,num_bins=10,normalization='none',
                           channel_layout='polarity_major',temporal_interpolation='none')
    times=torch.tensor([-1,0,3999,4000,39999,40000])
    result=voxel(torch.zeros(6),torch.zeros(6),times,torch.tensor([1,1,1,0,0,1]),start_time=0,end_time=40000)
    assert result.sum()==4
    assert result[0,0,0]==2
    assert result[11,0,0]==1
    assert result[19,0,0]==1
    empty=torch.empty(0)
    assert voxel(empty,empty,empty,empty,start_time=0,end_time=40000).sum()==0
    linear=EventVoxelizer(normalization='none')
    assert 'temporal_interpolation' not in event_representation_metadata(linear)
    assert event_representation_metadata(voxel)['window_boundary']=='[start,end)'


@pytest.mark.parametrize('width',[160,320])
def test_sum_pool_matches_direct_floor_binning(width):
    height=width*3//4
    generator=torch.Generator().manual_seed(7)
    x=torch.randint(640,(300,),generator=generator)
    y=torch.randint(480,(300,),generator=generator)
    t=torch.randint(40000,(300,),generator=generator)
    p=torch.randint(2,(300,),generator=generator)
    voxel=EventVoxelizer(normalization='none',channel_layout='polarity_major',temporal_interpolation='none')
    counts=voxel(x,y,t,p,start_time=0,end_time=40000)
    transform=PairedSequenceTransform(height=height,width=width,output_height=((height+15)//16)*16,
                                      event_resize_mode='sum_pool')
    actual,_=transform(counts[None],None)
    expected=torch.zeros(20,height,width)
    # Independently emulate OpenEB's direct integer coordinate and time bin rules.
    for xx,yy,tt,pp in zip(x.tolist(),y.tolist(),t.tolist(),p.tolist()):
        expected[(0 if pp else 10)+tt*10//40000,yy*height//480,xx*width//640]+=1
    pad=(actual.shape[-2]-height)//2
    torch.testing.assert_close(actual[0,:,pad:pad+height],expected,rtol=0,atol=0)
    assert actual.sum()==300
