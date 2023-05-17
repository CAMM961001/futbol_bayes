data {
  int<lower=0> N;          // Número de partidos
  int<lower=0> K;          // Número de equipos
  int<lower=0> T;          // Número de momentos en el tiempo
  int<lower=1,upper=K> team1[N];  // Índices de equipos locales
  int<lower=1,upper=K> team2[N];  // Índices de equipos visitantes
  int<lower=0> y1[N];      // Goles equipo local
  int<lower=0> y2[N];      // Goles equipo visitante
}

parameters {
  real home;
  real intercept;
  matrix[K, T] attack_raw;
  matrix[K, T] defence_raw;
  real<lower=0> sigma_attack;
  real<lower=0> sigma_defence;
}

transformed parameters {
  matrix[K, T] attack;
  matrix[K, T] defence;

  for (k in 1:K) {
    attack[k] = attack_raw[k] - mean(attack_raw[k]);
    defence[k] = defence_raw[k] - mean(defence_raw[k]);
  }
}

model {
  vector[N] theta1;
  vector[N] theta2;

  // Priors
  intercept ~ normal(0, 1);
  home ~ normal(0, 1);
  sigma_attack ~ cauchy(0, 3);
  sigma_defence ~ cauchy(0, 3);
  
  // Evolution equations
  for (k in 1:K) {
    for (t in 2:T) {
      attack_raw[k, t] ~ normal(attack_raw[k, t-1], sigma_attack);
      defence_raw[k, t] ~ normal(defence_raw[k, t-1], sigma_defence);
    }
    // Identificability
    sum(attack[k]) ~ normal(0, 0.01);
    sum(defence[k]) ~ normal(0, 0.01);
  }

  for (n in 1:N) {
    theta1[n] = home + attack[team1[n], n] - defence[team2[n], n] + intercept;
    theta2[n] = attack[team2[n], n] - defence[team1[n], n] + intercept;
    // Likelihood
    y1[n] ~ poisson_log(theta1[n]);
    y2[n] ~ poisson_log(theta2[n]);
  }
}

//generated quantities {
  //int y1_sim[N];
  //int y2_sim[N];
  //vector[N] theta1_sim;
  //vector[N] theta2_sim;

  //for (n in 1:N) {
    //theta1_sim[n] = home + attack[team1[n], n] - defence[team2[n], n] + intercept;
    //theta2_sim[n] = attack[team2[n], n] - defence[team1[n], n] + intercept;

    //y1_sim[n] = poisson_log_rng(theta1_sim[n]);
    //y2_sim[n] = poisson_log_rng(theta2_sim[n]);
//}
//}