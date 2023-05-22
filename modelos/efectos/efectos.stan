/*Este es un modelo para hacer inferencia bayesiana en el 
número de goles que puede anotar una selección de fútbol en 
competencias mundiales.

La construcción del código está basada en el modelo realizado
para la liga Premier de fútbol de clubes, realizada por el 
equipo de Stan, mismo que se explica en el siguiente tutorial:
https://www.youtube.com/watch?v=dNZQrcAjgXQ*/

data {
    // Número de partidos
    int <lower=0> n_partidos;
    
    // Número de equipos
    int <lower=0> n_equipos;
    
    // Indicadores de equipo
    array[n_partidos] int <lower=1, upper=n_equipos> locales;
    array[n_partidos] int <lower=1, upper=n_equipos> visitantes;
    
    // Datos de goles
    array[n_partidos] int <lower=0> goles_local;
    array[n_partidos] int <lower=0> goles_visita;
}

parameters {
    // Parámetro de equipo local
    real boost_local;

    // Parámetros de ataque y defensa base
    vector[n_equipos] base_ataque;
    vector[n_equipos] base_defensa;
}

transformed parameters {
    // Restricción de identificabilidad
    vector[n_equipos] ataque = base_ataque - mean(base_ataque);
    vector[n_equipos] defensa = base_defensa - mean(base_defensa);

    // Parámetros de Poisson
    vector[n_partidos] lambda_local;
    vector[n_partidos] lambda_visita;

    // Función liga
    lambda_local = boost_local + ataque[locales] - defensa[visitantes];
    lambda_visita = ataque[visitantes] - defensa[locales];
}

model {
    // Inicial de factor de equipo local
    boost_local ~ normal(0, 1);

    // Iniciales de ataque y defensa base
    base_ataque ~ normal(0, 0.0001);
    base_defensa ~ normal(0, 0.0001);

    // Restricción de identificabilidad
    //sum(ataque) ~ normal(0, 0.0001);
    //sum(defensa) ~ normal(0, 0.0001);
    
    // Verosimilitud
    goles_local ~ poisson_log(lambda_local);
    goles_visita ~ poisson_log(lambda_visita);
}

generated quantities {
    // Declaración de parámetros para simulaciones de la posterior
    vector[n_partidos] pred_lambda_local;
    vector[n_partidos] pred_lambda_visita;
    array[n_partidos] real sims_local;
    array[n_partidos] real sims_visita;

    // Función liga
    pred_lambda_local = boost_local + ataque[locales] - defensa[visitantes];
    pred_lambda_visita = ataque[visitantes] - defensa[locales];

    // Verosimilitud
    sims_local = poisson_log_rng(pred_lambda_local);
    sims_visita = poisson_log_rng(pred_lambda_visita);

    /*// Declaración de parámetros para log-verosimilitud
    array[n_partidos] real logver_local;
    array[n_partidos] real logver_visita;

    // Log-verosimilitud para métricas de desempeño
    for (partido in 1:n_partidos){
        logver_local[partido] = poisson_lpmf(goles_local[partido] | exp(lambda_local[partido]));
        logver_visita[partido] = poisson_lpmf(goles_visita[partido] | exp(lambda_visita[partido]));
    }*/
}
