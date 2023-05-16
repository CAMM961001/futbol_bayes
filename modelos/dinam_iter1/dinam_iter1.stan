data {
    int <lower=0> n_matches;
    vector[n_matches] goals_home;
    vector[n_matches] goals_away;
}

parameters {
    vector[n_matches] intercept;
    vector[n_matches] home_boost;
    vector[n_matches] attack;
    vector[n_matches] deffence;

    real <lower=0> sigma_attack;
    real <lower=0> sigma_deffence;
}

transformed parameters {
    vector[n_matches] lambda_home;
    vector[n_matches] lambda_away;

    for (match in 1:n_matches){
        lambda_home[match] = exp(intercept[match] + home_boost[match] + attack[match] - deffence[match]);
        lambda_away[match] = exp(intercept[match] + attack[match] - deffence[match]);
    }   
}

model { 
    intercept ~ normal(0, 1);
    home_boost ~ normal(0, 1);
    attack ~ normal(0, sigma_attack);
    deffence ~ normal(0, sigma_deffence);
    
    sigma_attack ~ gamma(5,5);
    sigma_deffence ~ gamma(5,5);
}

generated quantities {
    // ... declarations ... statements ...
}
