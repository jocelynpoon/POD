import numpy as np
from matplotlib import pyplot as plt

################################################################################################
#m = maximum camber ratio
#p = location of maximum camber
#t = maximum thickness ratio

x_rot = 0.25 #rotation about quarter chord
y_rot = 0

def naca(m, p, t, num_pts=100):
    beta = np.linspace(0, np.pi, num_pts)
    x = 0.5 * (1 - np.cos(beta))
    y_t = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)

    #camber
    y_c = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m != 0 and p != 0:
        loc1 = x < p
        loc2 = x >= p
        y_c[loc1] = m/p**2 * (2*p*x[loc1] - x[loc1]**2)
        y_c[loc2] = m/(1-p)**2 * ((1-2*p) + 2*p*x[loc2] - x[loc2]**2)
        dyc_dx[loc1] = 2*m/p**2 * (p-x[loc1])
        dyc_dx[loc2] = 2*m/(1-p)**2 * (p-x[loc2])
    theta = np.arctan(dyc_dx) #local angle of camber line
    x_upper = x-y_t*np.sin(theta) #surface coords
    y_upper = y_c+y_t*np.cos(theta)
    x_lower = x+y_t*np.sin(theta)
    y_lower = y_c-y_t*np.cos(theta)
    X = np.concatenate((x_upper[::-1], x_lower[1:]))
    Y = np.concatenate((y_upper[::-1], y_lower[1:]))
    return X, Y

def rotation(x_rot, y_rot, x, angle, y):
    x_new = x_rot+(x-x_rot)*np.cos(angle)-(y-y_rot)*np.sin(angle)
    y_new = y_rot+(x-x_rot)*np.sin(angle)+(y-y_rot)*np.cos(angle)
    return x_new, y_new

################################################################################################
#SNAPSHOTS varying thickness and camber with rotation

num_pts = 100
p = 0.4
thickness_values = np.linspace(0.08, 0.18, 6)
camber_values = np.linspace(0.00, 0.05, 6)
angle_values = np.linspace(-2*np.pi/180,2*np.pi/180, 6)
parameter_combinations = [(t, m, c) for t in thickness_values for m in camber_values for c in angle_values]
num_snapshots = len(parameter_combinations) #36
num_nodes = 2*num_pts-1 #same LE point
snapshot_matrix = np.zeros((2*num_nodes, num_snapshots))

parameters = np.zeros((num_snapshots, 3))
for i, (t, m, c) in enumerate(parameter_combinations):
    x, y = naca(m=m, p=p, t=t, num_pts=num_pts)
    a, b = rotation(x_rot, y_rot, x, c, y)
    snapshot_matrix[:, i] = np.concatenate((a, b))
    parameters[i, :] = [t, m, c]
print("Snapshot matrix shape:", snapshot_matrix.shape)


################################################################################################
#plot training geoms

plt.figure(figsize=(9, 5))
for i in range(0, 216, 4):
    t, m, c = parameter_combinations[i]
    x, y = naca(m=m, p=p, t=t, num_pts=num_pts)
    a, b = rotation(x_rot, y_rot, x, c, y)
    #plt.plot(a, b, label=f"t={t}, m={m}, c={c}")
    plt.plot(a, b, label=f"t={t:.3f}, m={m:.3f}")


plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Airfoil Geometries")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()

################################################################################################
# POD mean geom and SVD

mean_geom = np.mean(snapshot_matrix, axis=1)
snapshot_matrix_centered = snapshot_matrix - mean_geom[:, None]
modes, Sigma, Vt = np.linalg.svd(snapshot_matrix_centered,full_matrices=False)
eigenvalues = Sigma**2
total_energy = np.sum(eigenvalues)
mode_energy = eigenvalues / total_energy
cumulative_energy = np.cumsum(mode_energy)

print("\nPOD energy:")
for i in range(min(10, len(mode_energy))):
    print(f"Mode {i+1}: {mode_energy[i]*100:.12f}% | "f"Cumulative: {cumulative_energy[i]*100:.12f}%" )

################################################################################################
# reconstruct training snapshot

train_index = 20
q_train = snapshot_matrix[:, train_index]
r = 2
U_r = modes[:, :r]
a_train = U_r.T @ (q_train-mean_geom) #project onto POD modes for reduced order coeffs
q_train_reconstructed = mean_geom + U_r @ a_train
train_error = np.linalg.norm(q_train-q_train_reconstructed)
print("\nTraining reconstruction error:", train_error)

################################################################################################
# Test unseen parameters
t_test = 0.135
m_test = 0.025
rot_test = 1*np.pi/180

x_test, y_test = naca(m=m_test,p=p,t=t_test,num_pts=num_pts)
x_test_rot, y_test_rot =rotation(x_rot, y_rot, x_test, rot_test, y_test)
test_vector = np.concatenate((x_test_rot, y_test_rot))

r = 4
U_r = modes[:, :r]
print("U_r", U_r)
a_test = U_r.T @ (test_vector-mean_geom)
#print("a_test ", a_test)
test_reconstructed = mean_geom + U_r @ a_test
error = np.linalg.norm(test_vector-test_reconstructed)
relative_error = error / np.linalg.norm(test_vector-mean_geom)

print("\nTest parameters:")
print("Thickness:", t_test)
print("Camber:", m_test)
print(f"Reconstruction L2 error using {r} modes: {error:.6e}")
print(f"Relative reconstruction error: {relative_error*100:.4f}%")


################################################################################################
# test and reconstructed airfoil

'''
x_reconstructed = test_reconstructed[:num_nodes]
y_reconstructed = test_reconstructed[num_nodes:]
#print("x_recon", x_reconstructed)

plt.figure(figsize=(9, 5))
plt.plot(x_test_rot, y_test_rot,'k-', linewidth=2,label="Original test geometry")
plt.plot(x_reconstructed, y_reconstructed,'r--', linewidth=2,label=f"POD reconstruction ({r} modes)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("POD Reconstruction with new Parameters")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
'''

x_recon = []
y_recon = []

# plot for different number of modes
for i in range(1,r):
    U_r = modes[:, :i]
    a_test = U_r.T @ (test_vector-mean_geom)
    #print("a_test ", a_test)
    test_reconstructed = mean_geom + U_r @ a_test
    error = np.linalg.norm(test_vector-test_reconstructed)
    x_reconstructed = test_reconstructed[:num_nodes]
    y_reconstructed = test_reconstructed[num_nodes:]
    x_recon.append(x_reconstructed)
    y_recon.append(y_reconstructed)
    #print("len", len(x_recon))
    plt.figure(figsize=(9, 5))
    plt.plot(x_test_rot, y_test_rot,'k-', linewidth=2,label="Original test geometry")
    plt.plot(x_recon[i-1], y_recon[i-1],'r--', linewidth=2,label=f"POD reconstruction ({i} modes)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("POD Reconstruction with new Parameters")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()


################################################################################################
# error vs num POD modes

errors = []
for r in range(1, len(Sigma) + 1):
    U_r = modes[:, :r]
    a_test = U_r.T @ (test_vector-mean_geom)
    test_reconstructed = mean_geom + U_r @ a_test
    error = np.linalg.norm(test_vector-test_reconstructed)
    relative_error = error/np.linalg.norm(test_vector-mean_geom)
    #errors.append(relative_error)
    errors.append(error)

plt.figure(figsize=(8, 5))
#plt.plot(np.arange(1, len(errors)+1),np.array(errors)*100,'o-')
plt.semilogy(np.arange(1, len(errors)+1),np.array(errors),'o-')
plt.xlabel("Number of POD modes")
plt.ylabel("L2 Error")
plt.title("POD Reconstruction Accuracy")
plt.grid(True)
plt.show()

################################################################################################
