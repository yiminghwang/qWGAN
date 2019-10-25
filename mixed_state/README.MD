### learning the mixed state by qWGAN

This file outlines the procedure to learn mixed states using the qWGAN framwork. Example settings are provided in the configuration file config_mixed.py

#### Preparation

1. Create a random mixed quantum state which is an ensemble of two pure states as our target state
``` python
input_state = list()
angle = np.random.randint(1,10,size=[cf.num_to_mix,cf.system_size,3])
for i in range(cf.num_to_mix):
    matrix = Identity(cf.system_size)
    for j in range(cf.system_size):
        row_i_mat = np.matmul(Z_Rotation(cf.system_size, j, np.pi * angle[i][j][2], False),
                              np.matmul(Y_Rotation(cf.system_size, j, np.pi * angle[i][j][1], False),
                                        X_Rotation(cf.system_size, j, np.pi * angle[i][j][0], False)))
        matrix = np.matmul(row_i_mat, matrix)
    state = np.matmul(matrix, zero_state)
    input_state.append(np.asmatrix(state))
prob_real = [0.2, 0.8]
real_state = getreal_denmat(cf,prob_real,input_state)
```
2. Define the Generator and Discriminator
* Create an instance of the generator
```python
gen = Generator(cf.system_size, cf.num_to_mix)
```
* Construct and assign a quantum circuit as the generator
``` python
gen.set_qcircuit(construct_qcircuit(qc_list_gen, cf.system_size))
```
* Create an instance of the Discriminator

Define the set of Hermitian matrices whose linear combination is used in the discriminator.

``` python
herm = [I, X, Y, Z]
    dis = Discriminator(herm, cf.system_size)
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
```
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
the fidelity curve and loss curve of 2 qubits mixed state learning task
![Alt text](./figure/2qubit_mixed_state_0.png)
