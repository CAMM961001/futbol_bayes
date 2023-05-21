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
    int <lower=1, upper=n_equipos> locales[n_partidos];
    int <lower=1, upper=n_equipos> visitantes[n_partidos];
    
    // Datos de goles
    int <lower=0> goles_local[n_partidos];
    int <lower=0> goles_visita[n_partidos];
}

parameters {
    // Parámetro de equipo local
    real boost_local;
    
    // Hiper-parámetros de ataque y denfesa
    real mu_ataque;
    real mu_defensa;
    real <lower=0> sigma_ataque;
    real <lower=0> sigma_defensa;

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
    
    /* Iniciales para hiper-parámetros de ataque y defensa.
    Al modificar estos parámetros se puede recuperar el
    comportamiento de los siguientes modelos:
    -   Modelo de efectos constantes:
        1. Se debe recuperar el cero en el hiper-parámetro
        de centralidad.
        2. Se debe tener desviación estándar muy pequeña
        en el hiper-parámetro de escala.
        
        Esto se puede conseguir con:
        -   mu_ataque ~ normal(0, 0.001)
        -   mu_defensa ~ normal(0, 0.001)
        -   sigma_defensa ~ gamma(1, 0.001)
        -   sigma_defensa ~ gamma(1, 0.001)

    -   Modelo de efectos independientes:
        1. Se debe recuperar el cero en el hiper-parámetro
        de centralidad.
        2. Se debe tener desviación estándar muy grande
        en el hiper-parámetro de escala.
        
        Esto se puede conseguir con:
        -   mu_ataque ~ normal(0, 0.001)
        -   mu_defensa ~ normal(0, 0.001)
        -   sigma_defensa ~ gamma(1, 100)
        -   sigma_defensa ~ gamma(1, 100)
    
    -   Modelo de efectos intercambiables:
        Explorar cualquier valor intermedio en las iniciales

    Sin embargo, la definición de los modelos de efectos
    constantes e independientes utilizando este enfoque
    es sumamente costoso, y esta la razón por la que se
    definen de forma independiente en otro modelo.*/
    mu_ataque ~ normal(0, 0.1);
    mu_defensa ~ normal(0, 0.1);
    sigma_ataque ~ gamma(1, 0.1);
    sigma_defensa ~ gamma(1, 0.1);

    // Iniciales de ataque y defensa base
    base_ataque ~ normal(mu_ataque, sigma_ataque);
    base_defensa ~ normal(mu_defensa, sigma_defensa);

    // Restricción de identificabilidad
    sum(ataque) ~ normal(0, 0.0001);
    sum(defensa) ~ normal(0, 0.0001);
    
    // Verosimilitud
    goles_local ~ poisson_log(lambda_local);
    goles_visita ~ poisson_log(lambda_visita);
}

generated quantities {
    // Simulaciones de la posterior
    vector[n_partidos] pred_lambda_local;
    vector[n_partidos] pred_lambda_visita;
    real sims_local[n_partidos];
    real sims_visita[n_partidos];

    pred_lambda_local = boost_local + ataque[locales] - defensa[visitantes];
    pred_lambda_visita = ataque[visitantes] - defensa[locales];

    sims_local = poisson_log_rng(pred_lambda_local);
    sims_visita = poisson_log_rng(pred_lambda_visita);
}
