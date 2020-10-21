from spfem_mfront import Plasticity
import numpy as np
import stripy

lx = .1; ly = .1; g = [0, -10.]
rho = 2650.; mu = np.tan(19.8/180*np.pi)
mater = {
    'rho':rho,
    'young':1.5e6,
    'poisson':0.3,
    'mu':mu,
    'coh':0.,
    'gs':1.e-4,
    'alpha':0.025,
    'eps0':1.0,
    'vz':.5,
    'c0':0.025,
    'chi_inf':0.08,
    'chi0':0.04,
}

space = ly / 50.
np.random.seed(12345)
mesh = stripy.cartesian_meshes.square_mesh([0., lx, 0., ly], space, space, 0.3)
pts = np.column_stack((mesh.x, mesh.y))
npts = len(pts)
print('space = %f'%space)
print('npts = %d'%npts)
charlen = 1.5 * space

t_inf = np.sqrt(ly / -g[1]) * 4.
sigv = mater['rho'] * g[1] * (ly - pts[:,1])
k0 = 1. - np.sin(np.arctan(mater['mu'])); print('k0 =',k0)
sigh = k0 * sigv
sigma = np.column_stack([sigh, sigv, sigh, np.zeros(npts)])
prob = Plasticity(pts,mater,sigma,gravity=g,charlen=charlen)
dt0 = prob.getTimeStepSize()
print('Initial time step size:',dt0,'t_inf: ',t_inf)
print('Time steps estimated:',int(t_inf / dt0))
sig = prob.getStress()
eps = prob.getStrain()
chi = prob.getCompactivity()
vp = prob.getPlasticStrain()
disp = prob.getNodalDisp()
vel = prob.getNodalVel()
prob.saveMeshVar(name_prefix='./result/step_0_',disp=disp,vel=vel,sig=sig,eps=eps,chi=chi,vp=vp)

t = 0.
left_pts = (pts[:,0] < 1.e-6)
left_msk = np.outer(left_pts, [True, False]).reshape(-1)
prob.initTimeStep(Dmsk=left_msk,FricBound=np.array([[[-lx/10., 0.],[lx*3., 0.]],]),mu=[mu,])

tstep = [t_inf/200*i for i in range(1,211)]
step = 0; rem = 0
while (t < t_inf):
    dt = prob.getTimeStepSize()
    t += dt
    du = prob.solve()
    if t >= tstep[step]:
        step += 1
        sig = prob.getStress()
        eps = prob.getStrain()
        chi = prob.getCompactivity()
        vp = prob.getPlasticStrain()
        disp = prob.getNodalDisp()
        vel = prob.getNodalVel()
        prob.saveMeshVar(name_prefix='./result/step_%d_'%step,disp=disp,vel=vel,sig=sig,eps=eps,chi=chi,vp=vp)
