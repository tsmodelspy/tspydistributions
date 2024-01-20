from pytest import approx
import tspydistributions.pdqr as pdqr
import tspydistributions.density as ds
import tspydistributions.estimation as es
import torch
import numpy as np

def test_std():
    d1 = pdqr.dstd(0., 0., 1., 5.)
    d2 = ds.dstd(torch.tensor([0.,1.,5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_dstd(torch.tensor([0.,1.,5.], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.], dtype = torch.float64), list([0,1,2]), torch.tensor([1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_ged():
    d1 = pdqr.dged(0., 0., 1., 5.)
    d2 = ds.dged(torch.tensor([0.,1.,5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_dged(torch.tensor([0.,1.,5.], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.], dtype = torch.float64), list([0,1,2]), torch.tensor([1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_sged():
    d1 = pdqr.dsged(0., 0., 1., 1.5, 5.)
    d2 = ds.dsged(torch.tensor([0.,1.,1.5, 5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_dsged(torch.tensor([0.,1.,1.5,5.], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.,1.], dtype = torch.float64), list([0,1,2,3]), torch.tensor([1.,1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_snorm():
    d1 = pdqr.dsnorm(0., 0., 1., 1.5)
    d2 = ds.dsnorm(torch.tensor([0.,1.,1.5], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_dsnorm(torch.tensor([0.,1.,1.5], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.], dtype = torch.float64), list([0,1,2]), torch.tensor([1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_sstd():
    d1 = pdqr.dsstd(0., 0., 1., 1.5, 5.)
    d2 = ds.dsstd(torch.tensor([0.,1.,1.5, 5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_dsstd(torch.tensor([0.,1.,1.5,5.], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.,1.], dtype = torch.float64), list([0,1,2,3]), torch.tensor([1.,1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_jsu():
    d1 = pdqr.djsu(0., 0., 1., 1.5, 5.)
    d2 = ds.djsu(torch.tensor([0.,1.,1.5, 5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.ad_djsu(torch.tensor([0.,1.,1.5,5.], dtype = torch.float64, requires_grad=True), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.,1.], dtype = torch.float64), list([0,1,2,3]), torch.tensor([1.,1.,1.,1.], dtype = torch.float64))[0])
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_sghst():
    d1 = pdqr.dsghst(0., 0., 1., 1.5, 5.)
    d2 = ds.dsghst(torch.tensor([0.,1.,1.5, 5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.fd_dsghst(torch.tensor([0.,1.,1.5,5.], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.,1.], dtype = torch.float64), list([0,1,2,3]), torch.tensor([1.,1.,1.,1.], dtype = torch.float64)))
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True

def test_sgh():
    d1 = pdqr.dsgh(0., 0., 1., 0.9, 5.,-0.5)
    d2 = ds.dsgh(torch.tensor([0.,1.,0.9, 5., -0.5], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64)).numpy()
    d3 = np.exp(-1. * es.fd_dsgh(torch.tensor([0.,1.,0.9,5.,-0.5], dtype = torch.float64), torch.tensor([0.], dtype = torch.float64), torch.tensor([1.,1.,1.,1.,1.], dtype = torch.float64), list([0,1,2,3,4]), torch.tensor([1.,1.,1.,1.,1.], dtype = torch.float64)))
    a1 = approx(d1[0], 0.000001) == d2[0]
    a2 = approx(d1[0], 0.000001) == d3
    assert all([a1, a2]) == True
