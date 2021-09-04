import numpy as np
from spfem_elasticity import Elasticity
import stripy

mater = {
    'rho':1.,
    'young':100.,
    'poisson':.0
}

lx = 10.
ly = 1.
np.random.seed(12345)
extent = [0.,lx,0.,ly]
space = .2
mesh = stripy.cartesian_meshes.square_mesh(extent,space,space,.5)
pts = np.column_stack((mesh.x,mesh.y))
print(pts)
print(len(pts),max(pts[:,0]),max(pts[:,1]))
charlen = 1.5*space

prob = Elasticity(pts=pts,mater=mater,charlen=charlen)
dt = prob.getTimeStepSize()
nsteps = int(4./dt)

v0_x = 0.1
v0 = np.outer(np.sin(np.pi / (2.*lx) * pts[:,0]),[v0_x,0.]).reshape(-1)
prob.setNodalVel(v0)

left_msk = np.outer(pts[:,0] == 0.,[True, True]).reshape(-1)
bottom_msk = np.outer(pts[:,1] == 0.,[False, True]).reshape(-1)
top_msk = np.outer(pts[:,1] == ly,[False, True]).reshape(-1)
dirichlet_msk = left_msk + bottom_msk + top_msk

prob.initTimeStep(Dmsk=dirichlet_msk)

step = 0
disp = prob.getNodalDisp()
vel = prob.getNodalVel()
prob.saveMeshVar(name_prefix='res/step_%d_'%step,disp=disp,vel=vel)
while (step < nsteps):
   prob.solve()
   step += 1
   if step%40 == 0:
      disp = prob.getNodalDisp()
      vel = prob.getNodalVel()
      prob.saveMeshVar(name_prefix='res/step_%d_'%step,disp=disp,vel=vel)
