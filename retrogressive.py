from SPFEM_MFront import spfem_mfront
import numpy as np
import pygmsh
import os

mater = {
    'rho':2000.,
    'young':1.e6,
    'poisson':.33,
    'tau0':2.e4,
    'taur':4.e3,
    'b':5.
}

lx = 25.; ly=5.; mesh_size = .12
with pygmsh.geo.Geometry() as geom:
    geom.add_polygon([[0.,0.],[lx,0.],[lx-ly,ly],[0.,ly]],mesh_size=mesh_size)
    mesh = geom.generate_mesh()
pts = mesh.points[:,:-1]
npts = len(pts)
print('npts = %d'%npts)

charlen = 1.5 * mesh_size
g = [0., -10.]
height = np.minimum(lx - pts[:,0], ly)
sigv = mater['rho'] * g[1] * (height - pts[:,1])
k0 = 0.6
sigh = k0 * sigv
sigma = np.column_stack([sigh, sigv, sigh, np.zeros(npts)])

prob = spfem_mfront(pts,mater,sigma=sigma,gravity=g,charlen=charlen)
dt0 = prob.getTimeStepSize()
t_inf = 5.
print('Initial time step size:',dt0,'t_inf: ',t_inf)
print('Time steps estimated:',int(t_inf / dt0))
sig = prob.getStress()
eps = prob.getStrain()
vp = prob.getPlasticStrain()
disp = prob.getNodalDisp()
vel = prob.getNodalVel()

if not os.path.isdir('result'):
    os.makedirs('result')
prob.saveMeshVar(name_prefix='./result/step_0_',disp=disp,vel=vel,sig=sig,eps=eps,vp=vp)

left_pts = (pts[:,0] < 1.e-6)
left_msk = np.outer(left_pts, [True, False]).reshape(-1)
bottom_pts = (pts[:,1] < 1.e-6) * (pts[:,0] < lx + 1.e-6)
bottom_msk = np.outer(bottom_pts, [True, True]).reshape(-1)

prob.initTimeStep(Dmsk=left_msk+bottom_msk,FricBound=np.array([[[0., 0.],[lx*3., 0.]],]),mu=[.3,])

tstep = [t_inf/100*i for i in range(1,111)]
t = 0; step = 0
while (t < t_inf):
    dt = prob.getTimeStepSize()
    t += dt
    prob.solve()
    if t >= tstep[step]:
        step += 1
        if (step > 9) and (dt < dt0/10.):
            prob.smoothPoints()
            print('Time step changes from %f to %f after smoothing'%(dt,prob.getTimeStepSize()))
        sig = prob.getStress()
        eps = prob.getStrain()
        vp = prob.getPlasticStrain().copy()
        disp = prob.getNodalDisp()
        vel = prob.getNodalVel()
        prob.saveMeshVar(name_prefix='./result/step_%d_'%step,disp=disp,vel=vel,sig=sig,eps=eps,vp=vp)
