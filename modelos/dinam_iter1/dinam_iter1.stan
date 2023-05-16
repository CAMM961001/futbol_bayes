data {
    // Número de partidos
    int <lower=0> n_partidos;
    // Número de equipos
    int <lower=0> n_equipos;
    // Indicadores de equipo
    int <lower=1, upper=n_equipos> locales[n_partidos];
    int <lower=1, upper=n_equipos> visitantes[n_partidos];
    // Datos de goles
    int <lower=0> goles_local[n_partidos];
    int <lower=0> goles_visita[n_partidos];
}

parameters {
    // Parámetros de aplicación global
    real boost_local;
    real intercepto;
    real <lower=0> sigma_ataque;
    real <lower=0> sigma_defensa;
    // Parámetros de equipo
    vector[n_equipos] base_ataque;
    vector[n_equipos] base_defensa;
}

transformed parameters {
    // Restricción de suma cero
    vector[n_equipos] ataque = base_ataque - mean(base_ataque);
    vector[n_equipos] defensa = base_defensa - mean(base_defensa);
}

model { 
    vector[n_partidos] lambda_local;
    vector[n_partidos] lambda_visita;

    for (idp_ in 1:n_partidos) {
        lambda_local[idp_] = intercepto + boost_local + ataque[locales[idp_]] - defensa[visitantes[idp_]];
        lambda_visita[idp_] = intercepto + ataque[visitantes[idp_]] - defensa[locales[idp_]];
    }

    // Iniciales
    intercepto ~ normal(0, 1);
    boost_local ~ normal(0, 1);
    base_ataque ~ normal(0, sigma_ataque);
    base_defensa ~ normal(0, sigma_defensa);
    sigma_ataque ~ cauchy(0, 3);
    sigma_defensa ~ cauchy(0, 3);

    // Restricción de suma cero
    sum(ataque) ~ normal(0, 0.01);
    sum(defensa) ~ normal(0, 0.01);

    // Verosimilitud
    goles_local ~ poisson_log(lambda_local);
    goles_visita ~ poisson_log(lambda_visita);
}

generated quantities {
    // ... declarations ... statements ...
}
