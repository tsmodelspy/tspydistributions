from pytest import approx
import tspydistributions.pdqr as dist

def test_sgh():
    p = dist.psgh(dist.qsgh(0.5, -0.25, 0.9, 0.95, 4.1), -0.25, 0.9, 0.95, 4.1)
    assert approx(p[0], 0.00000001) == 0.5

def test_sghst():
    p = dist.psghst(dist.qsghst(0.5, -0.25, 0.9, 2, 4.1), -0.25, 0.9, 2, 4.1)
    assert approx(p[0], 0.000001) == 0.5

def test_jsu():
    p = dist.pjsu(dist.qjsu(0.5, -0.25, 0.95, 0.5, 0.5), -0.25, 0.95, 0.5, 0.5)
    assert approx(p[0], 0.00000001) == 0.5

def test_sged():
    p = dist.psged(dist.qsged(0.5, -0.25, 0.95, 0.5, 5), -0.25, 0.95, 0.5, 5)
    print(p)
    assert approx(p[0], 0.00000001) == 0.5

def test_norm():
    p = dist.pnorm(dist.qnorm(0.5, 0., 1.), 0., 1.)
    assert approx(p[0], 0.00000001) == 0.5 

def test_std():
    p = dist.pstd(dist.qstd(0.5, 0., 1.,5.), 0., 1., 5.)
    assert approx(p[0], 0.00000001) == 0.5 

def test_snorm():
    p = dist.psnorm(dist.qsnorm(0.5, -0.25, 0.95, 0.5), -0.25, 0.95, 0.5)
    assert approx(p[0], 0.00000001) == 0.5 

def test_sstd():
    p = dist.psstd(dist.qsstd(0.5, -0.25, 0.95, 0.5, 2.2), -0.25, 0.95, 0.5, 2.2)
    assert approx(p[0], 0.00000001) == 0.5
