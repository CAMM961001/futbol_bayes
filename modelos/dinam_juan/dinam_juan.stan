data {
  int<lower=0> N;          // Número de partidos
  int<lower=0> K;          // Número de equipos
  int<lower=1,upper=K> team1[N];  // Índices de equipos locales
  int<lower=1,upper=K> team2[N];  // Índices de equipos visitantes
  int<lower=0> y1[N];      // Goles equipo local
  int<lower=0> y2[N];      // Goles equipo visitante
}

parameters {
  real home;
  real intercept;
  vector[K] attack_raw;
  vector[K] defence_raw;
  real<lower=0> sigma_attack;
  real<lower=0> sigma_defence;
}

transformed parameters {
  vector[K] attack = attack_raw - mean(attack_raw);
  vector[K] defence = defence_raw - mean(defence_raw);
}

model {
  vector[N] theta1;
  vector[N] theta2;

  for (n in 1:N) {
    theta1[n] = home + attack[team1[n]] - defence[team2[n]] + intercept;
    theta2[n] = attack[team2[n]] - defence[team1[n]] + intercept;
  }

  // Priors
  intercept ~ normal(0, 1);
  home ~ normal(0, 1);
  attack_raw ~ normal(0, sigma_attack);
  defence_raw ~ normal(0, sigma_defence);
  sigma_attack ~ cauchy(0, 3);
  sigma_defence ~ cauchy(0, 3);

  // Identificabilidad
  sum(attack) ~ normal(0, 0.01);
  sum(defence) ~ normal(0, 0.01);

  // Likelihood
  y1 ~ poisson_log(theta1);
  y2 ~ poisson_log(theta2);
}
