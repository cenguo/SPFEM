import numpy as np
from spfem_elasticity import Elasticity
import pygmsh

mater = {
    'rho':1000.,
    'young':4.2e6,
    'poisson':.3
}

beta = np.pi / 3.
rad = 1.6; mesh_size = .2; g = np.array([np.sin(beta), -np.cos(beta)]) * 10.
print(g)
with pygmsh.geo.Geometry() as geom:
    geom.add_circle([0.,0.],rad,mesh_size=mesh_size)
    mesh = geom.generate_mesh()
pts = mesh.points[:,:-1]
npts = len(pts)
print('npts = %d'%npts)
charlen = 1.5 * mesh_size

prob = Elasticity(pts=pts,mater=mater,gravity=g,charlen=charlen)
dt = prob.getTimeStepSize()

center_pts = np.isclose(pts[:,0], 0.) * np.isclose(pts[:,1], 0.)
print(np.where(center_pts == True))

prob.initTimeStep(FricBound=np.array([[[-rad,-rad],[rad*15,-rad]],]),mu=[.3,])

step = 0
disp = prob.getNodalDisp()
prob.saveMeshVar(name_prefix='res/step_%d_'%step,disp=disp)
nstep = int(2./dt)
while (step < nstep):
   step += 1
   prob.solve()
   if step%200 == 0:
      disp = prob.getNodalDisp()
      prob.saveMeshVar(name_prefix='res/step_%d_'%step,disp=disp)
