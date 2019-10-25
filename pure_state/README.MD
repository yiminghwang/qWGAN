### Learning pure states using qWGAN

This document shows how the quantum Wasserstein GAN can be used to learn pure states.
 An example of the settings in configuration file '''config_pure.py'''.

#### Preparation

1. Create a random quantum state as our target state:

We use three direction rotation gates with random angles applied on each qubit to generate the target pure state.
``` python
angle = np.random.randint(1, 10, size=[size, 3])
matrix = Identity(size)
for j in range(size):
    row_i_mat = np.matmul(Z_Rotation(size, j, np.pi / angle[j][2], False),np.matmul(Y_Rotation(size, j, np.pi / angle[j][1], False),X_Rotation(size, j, np.pi / angle[j][0], False)))
matrix = np.matmul(row_i_mat, matrix)
real_state = np.matmul(matrix, zero_state)
```
2. Define the Generator and Discriminator
* Create an instance of the generator
```python
gen = Generator(size)
```
* Construct and assign the quantum circuit as the generator
``` python
gen.set_qcircuit(construct_qcircuit(qc_gen,size))
```
* Create an instance of the Discriminator

We define the fixed Hermitian matrices whose linear combination forms the Hermitian
``` python
herm = [I, X, Y, Z]
dis = Discriminator(herm, size)
```

#### Training
Alternately update the parameters of Generator and Discriminator until the fidelity between the generated quantum state and target state converges to 1.
``` python
gen.update_gen(dis,real_state)
```
``` python
dis.update_dis(gen,real_state)
```

#### Save the model and plot the figure
After training, the generator and discriminator weights can be saved
``` python
save_model(gen, cf.model_gen_path)
save_model(dis, cf.model_gen_path)
```
and the training loss and fidelity curves can be plotted
``` python
plt_fidelity_vs_iter(fidelities, losses, cf)
```

#### Output
the fidelity curve and loss curve for learning a 3 qubit pure state
![Alt text](./figure/[3]qubit_pure_state_0.png)
