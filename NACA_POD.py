import numpy as np
from matplotlib import pyplot as plt

################################################################################################
# TEST GEOMETRY AND ROTATION

def naca_0012(t):
    num_pts = 100
    beta = np.linspace(0,2*np.pi, num_pts)
    x = 0.5*(1-np.cos(beta))
    y=np.zeros_like(x)
    y_t = 5*t*(0.2969*np.sqrt(x)-0.1260*x-0.3516*x**2+0.2843*x**3-0.1015*x**4)
    y[beta <= np.pi] = -y_t[beta <= np.pi]
    y[beta > np.pi] = y_t[beta > np.pi]
    return x, y

t = .15
x, y = naca_0012(t)

def rotation(x_rot, y_rot, x, angle, y):
    x_new = x_rot+(x-x_rot)*np.cos(angle)-(y-y_rot)*np.sin(angle)
    y_new = y_rot+(x-x_rot)*np.sin(angle)+(y-y_rot)*np.cos(angle)
    return x_new, y_new

x_rot = 0.25
y_rot = 0
angle=1*np.pi/180
rot_x, rot_y = rotation(x_rot, y_rot, x, angle, y)
plt.figure(figsize=(8, 6))
plt.plot(rot_x, rot_y, marker='o', markersize=6, linestyle='-', label = "rotated")
plt.plot(x, y, marker='o', markersize=6, linestyle='-', label = "original")
plt.title("NACA 0012 shape")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()

########################################################################################################################
# POD

num_nodes = 100
angles= np.linspace(-5*np.pi/180,5*np.pi/180,20)
snapshot_matrix = np.zeros((2*num_nodes,len(angles))) #rows are x coords and y coords, columns are snapshots
print(snapshot_matrix.shape)
mean_geom = np.zeros(2*num_nodes)
#print(mean_geom)
for i in range(len(angles)):
    rot_x, rot_y = rotation(x_rot, y_rot, x, angles[i], y)
    snapshot_matrix[:,i] = np.concatenate((rot_x, rot_y))
    mean_geom += snapshot_matrix[:,i]

mean_geom/=len(angles)
print(mean_geom)

#mean center the snapshot matrix
for i in range(len(angles)):
    snapshot_matrix[:,i] -= mean_geom

#Compute SVD

#snapshot_covariance = snapshot_matrix.T @ snapshot_matrix
# do twisting snapshots represent an entire wing with twist?

modes, Sigma, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
eigenvalues = Sigma**2
total_energy = np.sum(eigenvalues)
cumulative_energy = np.cumsum(eigenvalues)/total_energy

print("Mode 1 Energy: ", cumulative_energy[0]*100)
print("Mode 2 Energy: ", cumulative_energy[1]*100)
mode_energy = eigenvalues/total_energy
print("Individual mode energy (%):")
print(mode_energy * 100)

# Reconstruct one of the original snapshots
q_train = snapshot_matrix[:, 10] + mean_geom

a_train = modes[:, :2].T @ (q_train - mean_geom)
q_train_reconstructed = mean_geom + modes[:, :2] @ a_train

train_error = np.linalg.norm(q_train - q_train_reconstructed)

print("Training snapshot error:", train_error)

################################################################################################
#TEST
test_angle = 3.33 * np.pi/180
x_test, y_test = rotation(x_rot, y_rot, x, test_angle, y)
test_vector_combined = np.concatenate((x_test,y_test)).flatten()
mean_geom = mean_geom.flatten()

r = 2 #first 2 modes
U_r = modes[:, :r]
print("modes ", U_r)

y_vector =test_vector_combined - mean_geom

#reduced order coefficients
a_coefficients = U_r.T@y_vector
#reconstruction
x_reconstructed_vector = mean_geom + U_r @ a_coefficients

#error
error = np.linalg.norm(test_vector_combined - x_reconstructed_vector)
print(f"\nReconstruction L2 Error using {r} modes: {error:.2e}")

print("Snapshot matrix:", snapshot_matrix.shape)
print("Mean geometry:", mean_geom.shape)
print("Test vector:", test_vector_combined.shape)
print("U_r:", U_r.shape)
print("Reconstructed:", x_reconstructed_vector.shape)