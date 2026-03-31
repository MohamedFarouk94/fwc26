from simulation import run_sim

rho = int(input('Enter Form Change Factor (0 - 10): ')) / 10
mu = int(input('Enter Shock Factor (0 - 10): ')) / 10

run_sim(rho=rho, mu=mu)
