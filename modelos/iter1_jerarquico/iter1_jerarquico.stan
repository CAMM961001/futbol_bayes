data {
    // Número de partidos
    int<lower=1> n_matches;
    // Goles por equipo
    array[n_matches] int<lower=0> goals_home;
    array[n_matches] int<lower=0> goals_away;
}

parameters {
    // Habilidad por equipo
    vector[n_matches] baseline;
    vector[n_matches] skill_home;
    vector[n_matches] skill_away;
    real mu_teams;
    real sigma_teams;
}

model {
    // Modelo de goles anotados
    goals_home ~ poisson_log(baseline + skill_home - skill_away);
    goals_away ~ poisson_log(baseline + skill_away - skill_home);
    // Modelo de habilidad por equipo
    skill_home ~ normal(mu_teams, pow(sigma_teams, 2));
    skill_away ~ normal(mu_teams, pow(sigma_teams, 2));
    // Parámetros de habilidad
    baseline ~ normal(0, pow(4, 2));
    mu_teams ~ normal(0, pow(4, 2));
    sigma_teams ~ uniform(0, 3);
}

generated quantities {
    array[n_matches] real sims = poisson_log_rng(baseline + skill_home - skill_away);
}