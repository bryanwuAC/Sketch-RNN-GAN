import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import PIL
import os
from hyperparameters import HyperParameters

hps = HyperParameters()


def apply_temperature_hyperparameter(distribution):
    new_distribution = np.log(distribution) / hps.tau
    new_distribution -= new_distribution.max()
    new_distribution = np.exp(new_distribution) / np.sum(np.exp(new_distribution))
    return new_distribution


def generate_position_from_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy):
    mean = [mu_x, mu_y]
    variance_x = sigma_x ** 2 * hps.tau
    variance_y = sigma_y ** 2 * hps.tau
    covariance = [[variance_x, rho_xy * sigma_x * sigma_y],
                  [rho_xy * sigma_x * sigma_y, variance_y]]
    generated_position = np.random.multivariate_normal(mean, covariance, 1)
    return generated_position[0][0], generated_position[0][1]


# Convert a mixture distribution to a 5-stroke data point
def distribution2stroke(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    pi = apply_temperature_hyperparameter(pi)
    selected_distribution = np.random.choice(hps.num_mixture, p=pi)

    q = apply_temperature_hyperparameter(q)
    selected_pen_state = np.random.choice(3, p=q)

    generated_x, generated_y = generate_position_from_bivariate_normal(mu_x[selected_distribution],
                                                                       mu_y[selected_distribution],
                                                                       sigma_x[selected_distribution],
                                                                       sigma_y[selected_distribution],
                                                                       rho_xy[selected_distribution])
    stroke = np.zeros(5)
    stroke[0] = generated_x
    stroke[1] = generated_y
    stroke[selected_pen_state + 2] = 1
    return stroke


def torch2numpy(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    pi = pi.data.cpu().numpy()
    mu_x = mu_x.data.cpu().numpy()
    mu_y = mu_y.data.cpu().numpy()
    sigma_x = sigma_x.data.cpu().numpy()
    sigma_y = sigma_y.data.cpu().numpy()
    rho_xy = rho_xy.data.cpu().numpy()
    q = q.data.cpu().numpy()
    return pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q


def generate_sequences_from_distributions(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q):
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = torch2numpy(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)

    result_sequences = []
    for i in range(hps.batch_size):
        sequence = []
        for j in range(pi.shape[0]):
            sequence.append(
                distribution2stroke(pi[j, i, :], mu_x[j, i, :], mu_y[j, i, :], sigma_x[j, i, :], sigma_y[j, i, :],
                                    rho_xy[j, i, :], q[j, i, :]))
        result_sequences.append(np.array(sequence))

    result_sequences = Variable(torch.from_numpy(np.stack(result_sequences, 1)).cuda().float())
    return result_sequences


def make_image(sequence, epoch, dataset_name):
    strokes = np.split(sequence, np.where(sequence[:, 3] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    os.makedirs(hps.output_dir.format(dataset_name), exist_ok=True)
    name = hps.output_path.format(dataset_name, str(epoch))
    pil_image.save(name, "JPEG")
    plt.close("all")


def generate_sequences_with_model(N_max, generator):
    z = Variable(torch.zeros(1, hps.latent_vector_length).cuda().float())
    current_stroke = Variable(torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda())
    hidden_cell = None
    sequence = []
    for i in range(N_max):
        inputs = torch.cat([current_stroke, z.unsqueeze(0)], 2)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = generator(inputs, z, hidden_cell)
        hidden_cell = (hidden, cell)

        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q = torch2numpy(pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q)
        generated_stroke = distribution2stroke(pi[0, 0, :], mu_x[0, 0, :], mu_y[0, 0, :], sigma_x[0, 0, :],
                                               sigma_y[0, 0, :], rho_xy[0, 0, :], q[0, 0, :])
        sequence.append(generated_stroke)
        # Stop if pen state vector is [x, y, 0, 0, 1]
        if generated_stroke[-1] == 1:
            break
        current_stroke = Variable(torch.from_numpy(generated_stroke).float().view(1, 1, -1).cuda())

    sequence = np.stack(sequence)
    sequence[:, 0] = np.cumsum(sequence[:, 0])
    sequence[:, 1] = np.cumsum(sequence[:, 1])
    print("Current image points:", sequence.shape[0])
    return sequence


def generate_image_with_model(N_max, generator, epoch, dataset_name):
    sequence = generate_sequences_with_model(N_max, generator)
    make_image(sequence, epoch, dataset_name)

