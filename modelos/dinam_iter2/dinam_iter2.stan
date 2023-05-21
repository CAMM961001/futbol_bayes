data {
    // Número de partidos
    int <lower=0> n_partidos;
    // Número de equipos
    int <lower=0> n_equipos;
    // Número de momentos en el tiempo
    int <lower=0> n_tiempo;

    // Vectores para iterar
    int <lower=1, upper=n_equipos> locales[n_partidos];
    int <lower=1, upper=n_equipos> visitantes[n_partidos];
    int <lower=1,upper=n_tiempo> tiempo[n_partidos];

    // Datos de goles
    int <lower=0> goles_local[n_partidos];
    int <lower=0> goles_visita[n_partidos];
}

parameters {
    // Parámetros de aplicación global
    real boost_local;
    real intercepto;
    matrix[n_equipos, n_tiempo] base_ataque;
    matrix[n_equipos, n_tiempo] base_defensa;
    real <lower=0> sigma_ataque;
    real <lower=0> sigma_defensa;
}

transformed parameters {
    matrix[n_equipos, n_tiempo] ataque;
    matrix[n_equipos, n_tiempo] defensa;
    
    // Restricción de suma cero
    for (idp_ in 1:n_equipos) {
        ataque[idp_] = base_ataque[idp_] - mean(base_ataque[idp_]);
        defensa[idp_] = base_defensa[idp_] - mean(base_defensa[idp_]);
    }
}

model { 
    vector[n_partidos] lambda_local;
    vector[n_partidos] lambda_visita;

    // Iniciales
    intercepto ~ normal(0, 1);
    boost_local ~ normal(0, 1);
    sigma_ataque ~ gamma(5, 5);
    sigma_defensa ~ gamma(5, 5);

    // Ecuación de evolución
    for (idp_ in 1:n_equipos) {
        for (idt_ in 2:n_tiempo) {
            base_ataque[idp_, idt_] ~ normal(base_ataque[idp_, idt_-1], sigma_ataque);
            base_defensa[idp_, idt_] ~ normal(base_defensa[idp_, idt_-1], sigma_defensa);
        }
        // Restricción de identificabilidad
        sum(ataque[idp_]) ~ normal(0, 0.01);
        sum(defensa[idp_]) ~ normal(0, 0.01);
    }

    for (idp_ in 1:n_partidos) {
        lambda_local[idp_] = boost_local + ataque[locales[idp_], tiempo[idp_]] - defensa[visitantes[idp_], tiempo[idp_]] + intercepto;
        lambda_visita[idp_] = ataque[visitantes[idp_], tiempo[idp_]] - defensa[locales[idp_], tiempo[idp_]] + intercepto;
        
        // Verosimilitud
        goles_local[idp_] ~ poisson_log(lambda_local[idp_]);
        goles_visita[idp_] ~ poisson_log(lambda_visita[idp_]);
    }
}
