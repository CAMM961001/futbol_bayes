
// Code generated by stanc v2.31.0
#include <stan/model/model_header.hpp>
namespace iter2_jera_model_namespace {

using stan::model::model_base_crtp;
using namespace stan::math;


stan::math::profile_map profiles__;
static constexpr std::array<const char*, 33> locations_array__ = 
{" (found before start of program)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 11, column 4 to column 35)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 12, column 4 to column 47)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 13, column 4 to column 47)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 14, column 4 to column 18)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 15, column 4 to column 21)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 20, column 4 to column 48)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 21, column 4 to column 48)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 24, column 8 to column 90)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 23, column 4 to line 24, column 90)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 27, column 8 to column 90)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 26, column 4 to line 27, column 90)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 44, column 4 to column 63)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 45, column 4 to column 63)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 32, column 4 to column 38)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 33, column 4 to column 38)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 35, column 4 to column 47)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 36, column 4 to column 47)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 38, column 4 to column 28)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 39, column 4 to column 28)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 40, column 4 to column 30)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 3, column 4 to column 27)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 5, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 5, column 4 to column 46)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 6, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 6, column 4 to column 46)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 11, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 12, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 13, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 20, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 21, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 44, column 10 to column 19)",
 " (in '/home/camm961001/Github/futbol_bayes/modelos/iter2_jera/iter2_jera.stan', line 45, column 10 to column 19)"};




class iter2_jera_model final : public model_base_crtp<iter2_jera_model> {

 private:
  int n_matches;
  std::vector<int> goals_home;
  std::vector<int> goals_away; 
  
 
 public:
  ~iter2_jera_model() { }
  
  inline std::string model_name() const final { return "iter2_jera_model"; }

  inline std::vector<std::string> model_compile_info() const noexcept {
    return std::vector<std::string>{"stanc_version = stanc3 v2.31.0", "stancflags = "};
  }
  
  
  iter2_jera_model(stan::io::var_context& context__,
                   unsigned int random_seed__ = 0,
                   std::ostream* pstream__ = nullptr) : model_base_crtp(0) {
    int current_statement__ = 0;
    using local_scalar_t__ = double ;
    boost::ecuyer1988 base_rng__ = 
        stan::services::util::create_rng(random_seed__, 0);
    (void) base_rng__;  // suppress unused var warning
    static constexpr const char* function__ = "iter2_jera_model_namespace::iter2_jera_model";
    (void) function__;  // suppress unused var warning
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      current_statement__ = 21;
      context__.validate_dims("data initialization","n_matches","int",
           std::vector<size_t>{});
      n_matches = std::numeric_limits<int>::min();
      
      
      current_statement__ = 21;
      n_matches = context__.vals_i("n_matches")[(1 - 1)];
      current_statement__ = 21;
      stan::math::check_greater_or_equal(function__, "n_matches", n_matches,
                                            1);
      current_statement__ = 22;
      stan::math::validate_non_negative_index("goals_home", "n_matches",
                                              n_matches);
      current_statement__ = 23;
      context__.validate_dims("data initialization","goals_home","int",
           std::vector<size_t>{static_cast<size_t>(n_matches)});
      goals_home = 
        std::vector<int>(n_matches, std::numeric_limits<int>::min());
      
      
      current_statement__ = 23;
      goals_home = context__.vals_i("goals_home");
      current_statement__ = 23;
      stan::math::check_greater_or_equal(function__, "goals_home",
                                            goals_home, 0);
      current_statement__ = 24;
      stan::math::validate_non_negative_index("goals_away", "n_matches",
                                              n_matches);
      current_statement__ = 25;
      context__.validate_dims("data initialization","goals_away","int",
           std::vector<size_t>{static_cast<size_t>(n_matches)});
      goals_away = 
        std::vector<int>(n_matches, std::numeric_limits<int>::min());
      
      
      current_statement__ = 25;
      goals_away = context__.vals_i("goals_away");
      current_statement__ = 25;
      stan::math::check_greater_or_equal(function__, "goals_away",
                                            goals_away, 0);
      current_statement__ = 26;
      stan::math::validate_non_negative_index("baseline", "n_matches",
                                              n_matches);
      current_statement__ = 27;
      stan::math::validate_non_negative_index("skill_home", "n_matches",
                                              n_matches);
      current_statement__ = 28;
      stan::math::validate_non_negative_index("skill_away", "n_matches",
                                              n_matches);
      current_statement__ = 29;
      stan::math::validate_non_negative_index("lambda_home", "n_matches",
                                              n_matches);
      current_statement__ = 30;
      stan::math::validate_non_negative_index("lambda_away", "n_matches",
                                              n_matches);
      current_statement__ = 31;
      stan::math::validate_non_negative_index("sims_home", "n_matches",
                                              n_matches);
      current_statement__ = 32;
      stan::math::validate_non_negative_index("sims_away", "n_matches",
                                              n_matches);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    num_params_r__ = n_matches + n_matches + n_matches + 1 + 1;
    
  }
  
  template <bool propto__, bool jacobian__ , typename VecR, typename VecI, 
  stan::require_vector_like_t<VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline stan::scalar_type_t<VecR> log_prob_impl(VecR& params_r__,
                                                 VecI& params_i__,
                                                 std::ostream* pstream__ = nullptr) const {
    using T__ = stan::scalar_type_t<VecR>;
    using local_scalar_t__ = T__;
    T__ lp__(0.0);
    stan::math::accumulator<T__> lp_accum__;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "iter2_jera_model_namespace::log_prob";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<local_scalar_t__> baseline =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      current_statement__ = 1;
      baseline = in__.template read<std::vector<local_scalar_t__>>(n_matches);
      std::vector<local_scalar_t__> skill_home =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      current_statement__ = 2;
      skill_home = in__.template read_constrain_lb<
                     std::vector<local_scalar_t__>, jacobian__>(0, lp__,
                     n_matches);
      std::vector<local_scalar_t__> skill_away =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      current_statement__ = 3;
      skill_away = in__.template read_constrain_lb<
                     std::vector<local_scalar_t__>, jacobian__>(0, lp__,
                     n_matches);
      local_scalar_t__ mu_teams = DUMMY_VAR__;
      current_statement__ = 4;
      mu_teams = in__.template read<local_scalar_t__>();
      local_scalar_t__ sigma_teams = DUMMY_VAR__;
      current_statement__ = 5;
      sigma_teams = in__.template read<local_scalar_t__>();
      std::vector<local_scalar_t__> lambda_home =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      std::vector<local_scalar_t__> lambda_away =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      current_statement__ = 9;
      for (int match = 1; match <= n_matches; ++match) {
        current_statement__ = 8;
        stan::model::assign(lambda_home,
          stan::math::exp(
            ((stan::model::rvalue(baseline, "baseline",
                stan::model::index_uni(match)) +
               stan::model::rvalue(skill_home, "skill_home",
                 stan::model::index_uni(match))) -
              stan::model::rvalue(skill_away, "skill_away",
                stan::model::index_uni(match)))),
          "assigning variable lambda_home", stan::model::index_uni(match));
      }
      current_statement__ = 11;
      for (int match = 1; match <= n_matches; ++match) {
        current_statement__ = 10;
        stan::model::assign(lambda_away,
          stan::math::exp(
            ((stan::model::rvalue(baseline, "baseline",
                stan::model::index_uni(match)) +
               stan::model::rvalue(skill_away, "skill_away",
                 stan::model::index_uni(match))) -
              stan::model::rvalue(skill_home, "skill_home",
                stan::model::index_uni(match)))),
          "assigning variable lambda_away", stan::model::index_uni(match));
      }
      current_statement__ = 6;
      stan::math::check_greater_or_equal(function__, "lambda_home",
                                            lambda_home, 0);
      current_statement__ = 7;
      stan::math::check_greater_or_equal(function__, "lambda_away",
                                            lambda_away, 0);
      {
        current_statement__ = 14;
        lp_accum__.add(
          stan::math::poisson_lpmf<propto__>(goals_home, lambda_home));
        current_statement__ = 15;
        lp_accum__.add(
          stan::math::poisson_lpmf<propto__>(goals_away, lambda_away));
        current_statement__ = 16;
        lp_accum__.add(
          stan::math::normal_lpdf<propto__>(skill_home, mu_teams,
            sigma_teams));
        current_statement__ = 17;
        lp_accum__.add(
          stan::math::normal_lpdf<propto__>(skill_away, mu_teams,
            sigma_teams));
        current_statement__ = 18;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(baseline, 0, 4));
        current_statement__ = 19;
        lp_accum__.add(stan::math::normal_lpdf<propto__>(mu_teams, 0, 4));
        current_statement__ = 20;
        lp_accum__.add(stan::math::gamma_lpdf<propto__>(sigma_teams, 5, 5));
      }
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    lp_accum__.add(lp__);
    return lp_accum__.sum();
    } // log_prob_impl() 
    
  template <typename RNG, typename VecR, typename VecI, typename VecVar, 
  stan::require_vector_like_vt<std::is_floating_point, VecR>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr, 
  stan::require_vector_vt<std::is_floating_point, VecVar>* = nullptr> 
  inline void write_array_impl(RNG& base_rng__, VecR& params_r__,
                               VecI& params_i__, VecVar& vars__,
                               const bool emit_transformed_parameters__ = true,
                               const bool emit_generated_quantities__ = true,
                               std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    static constexpr bool propto__ = true;
    (void) propto__;
    double lp__ = 0.0;
    (void) lp__;  // dummy to suppress unused var warning
    int current_statement__ = 0; 
    stan::math::accumulator<double> lp_accum__;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    constexpr bool jacobian__ = false;
    (void) DUMMY_VAR__;  // suppress unused var warning
    static constexpr const char* function__ = "iter2_jera_model_namespace::write_array";
    (void) function__;  // suppress unused var warning
    
    try {
      std::vector<double> baseline =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 1;
      baseline = in__.template read<std::vector<local_scalar_t__>>(n_matches);
      std::vector<double> skill_home =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 2;
      skill_home = in__.template read_constrain_lb<
                     std::vector<local_scalar_t__>, jacobian__>(0, lp__,
                     n_matches);
      std::vector<double> skill_away =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 3;
      skill_away = in__.template read_constrain_lb<
                     std::vector<local_scalar_t__>, jacobian__>(0, lp__,
                     n_matches);
      double mu_teams = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 4;
      mu_teams = in__.template read<local_scalar_t__>();
      double sigma_teams = std::numeric_limits<double>::quiet_NaN();
      current_statement__ = 5;
      sigma_teams = in__.template read<local_scalar_t__>();
      std::vector<double> lambda_home =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      std::vector<double> lambda_away =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      out__.write(baseline);
      out__.write(skill_home);
      out__.write(skill_away);
      out__.write(mu_teams);
      out__.write(sigma_teams);
      if (stan::math::logical_negation((stan::math::primitive_value(
            emit_transformed_parameters__) || stan::math::primitive_value(
            emit_generated_quantities__)))) {
        return ;
      } 
      current_statement__ = 9;
      for (int match = 1; match <= n_matches; ++match) {
        current_statement__ = 8;
        stan::model::assign(lambda_home,
          stan::math::exp(
            ((stan::model::rvalue(baseline, "baseline",
                stan::model::index_uni(match)) +
               stan::model::rvalue(skill_home, "skill_home",
                 stan::model::index_uni(match))) -
              stan::model::rvalue(skill_away, "skill_away",
                stan::model::index_uni(match)))),
          "assigning variable lambda_home", stan::model::index_uni(match));
      }
      current_statement__ = 11;
      for (int match = 1; match <= n_matches; ++match) {
        current_statement__ = 10;
        stan::model::assign(lambda_away,
          stan::math::exp(
            ((stan::model::rvalue(baseline, "baseline",
                stan::model::index_uni(match)) +
               stan::model::rvalue(skill_away, "skill_away",
                 stan::model::index_uni(match))) -
              stan::model::rvalue(skill_home, "skill_home",
                stan::model::index_uni(match)))),
          "assigning variable lambda_away", stan::model::index_uni(match));
      }
      current_statement__ = 6;
      stan::math::check_greater_or_equal(function__, "lambda_home",
                                            lambda_home, 0);
      current_statement__ = 7;
      stan::math::check_greater_or_equal(function__, "lambda_away",
                                            lambda_away, 0);
      if (emit_transformed_parameters__) {
        out__.write(lambda_home);
        out__.write(lambda_away);
      } 
      if (stan::math::logical_negation(emit_generated_quantities__)) {
        return ;
      } 
      std::vector<double> sims_home =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 12;
      stan::model::assign(sims_home,
        stan::math::poisson_rng(lambda_home, base_rng__),
        "assigning variable sims_home");
      std::vector<double> sims_away =
         std::vector<double>(n_matches, 
           std::numeric_limits<double>::quiet_NaN());
      current_statement__ = 13;
      stan::model::assign(sims_away,
        stan::math::poisson_rng(lambda_away, base_rng__),
        "assigning variable sims_away");
      out__.write(sims_home);
      out__.write(sims_away);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // write_array_impl() 
    
  template <typename VecVar, typename VecI, 
  stan::require_vector_t<VecVar>* = nullptr, 
  stan::require_vector_like_vt<std::is_integral, VecI>* = nullptr> 
  inline void transform_inits_impl(VecVar& params_r__, VecI& params_i__,
                                   VecVar& vars__,
                                   std::ostream* pstream__ = nullptr) const {
    using local_scalar_t__ = double;
    stan::io::deserializer<local_scalar_t__> in__(params_r__, params_i__);
    stan::io::serializer<local_scalar_t__> out__(vars__);
    int current_statement__ = 0;
    local_scalar_t__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());
    
    try {
      int pos__ = std::numeric_limits<int>::min();
      pos__ = 1;
      std::vector<local_scalar_t__> baseline =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        baseline[(sym1__ - 1)] = in__.read<local_scalar_t__>();
      }
      out__.write(baseline);
      std::vector<local_scalar_t__> skill_home =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        skill_home[(sym1__ - 1)] = in__.read<local_scalar_t__>();
      }
      out__.write_free_lb(0, skill_home);
      std::vector<local_scalar_t__> skill_away =
         std::vector<local_scalar_t__>(n_matches, DUMMY_VAR__);
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        skill_away[(sym1__ - 1)] = in__.read<local_scalar_t__>();
      }
      out__.write_free_lb(0, skill_away);
      local_scalar_t__ mu_teams = DUMMY_VAR__;
      mu_teams = in__.read<local_scalar_t__>();
      out__.write(mu_teams);
      local_scalar_t__ sigma_teams = DUMMY_VAR__;
      sigma_teams = in__.read<local_scalar_t__>();
      out__.write(sigma_teams);
    } catch (const std::exception& e) {
      stan::lang::rethrow_located(e, locations_array__[current_statement__]);
    }
    } // transform_inits_impl() 
    
  inline void get_param_names(std::vector<std::string>& names__) const {
    
    names__ = std::vector<std::string>{"baseline", "skill_home",
      "skill_away", "mu_teams", "sigma_teams", "lambda_home", "lambda_away",
      "sims_home", "sims_away"};
    
    } // get_param_names() 
    
  inline void get_dims(std::vector<std::vector<size_t>>& dimss__) const {
    
    dimss__ = std::vector<std::vector<size_t>>{std::vector<size_t>{
                                                                   static_cast<size_t>(n_matches)
                                                                   },
      std::vector<size_t>{static_cast<size_t>(n_matches)},
      std::vector<size_t>{static_cast<size_t>(n_matches)},
      std::vector<size_t>{}, std::vector<size_t>{},
      std::vector<size_t>{static_cast<size_t>(n_matches)},
      std::vector<size_t>{static_cast<size_t>(n_matches)},
      std::vector<size_t>{static_cast<size_t>(n_matches)},
      std::vector<size_t>{static_cast<size_t>(n_matches)}};
    
    } // get_dims() 
    
  inline void constrained_param_names(
                                      std::vector<std::string>& param_names__,
                                      bool emit_transformed_parameters__ = true,
                                      bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "baseline" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "skill_home" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "skill_away" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "mu_teams");
    param_names__.emplace_back(std::string() + "sigma_teams");
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "lambda_home" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "lambda_away" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "sims_home" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "sims_away" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // constrained_param_names() 
    
  inline void unconstrained_param_names(
                                        std::vector<std::string>& param_names__,
                                        bool emit_transformed_parameters__ = true,
                                        bool emit_generated_quantities__ = true) const
    final {
    
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "baseline" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "skill_home" + '.' + std::to_string(sym1__));
      } 
    }
    for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
      {
        param_names__.emplace_back(std::string() + "skill_away" + '.' + std::to_string(sym1__));
      } 
    }
    param_names__.emplace_back(std::string() + "mu_teams");
    param_names__.emplace_back(std::string() + "sigma_teams");
    if (emit_transformed_parameters__) {
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "lambda_home" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "lambda_away" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    if (emit_generated_quantities__) {
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "sims_home" + '.' + std::to_string(sym1__));
        } 
      }
      for (int sym1__ = 1; sym1__ <= n_matches; ++sym1__) {
        {
          param_names__.emplace_back(std::string() + "sims_away" + '.' + std::to_string(sym1__));
        } 
      }
    }
    
    } // unconstrained_param_names() 
    
  inline std::string get_constrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"baseline\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"skill_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"skill_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"mu_teams\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_teams\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"lambda_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"lambda_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"sims_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"},{\"name\":\"sims_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_constrained_sizedtypes() 
    
  inline std::string get_unconstrained_sizedtypes() const {
    
    return std::string("[{\"name\":\"baseline\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"skill_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"skill_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"parameters\"},{\"name\":\"mu_teams\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"sigma_teams\",\"type\":{\"name\":\"real\"},\"block\":\"parameters\"},{\"name\":\"lambda_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"lambda_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"transformed_parameters\"},{\"name\":\"sims_home\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"},{\"name\":\"sims_away\",\"type\":{\"name\":\"array\",\"length\":" + std::to_string(n_matches) + ",\"element_type\":{\"name\":\"real\"}},\"block\":\"generated_quantities\"}]");
    
    } // get_unconstrained_sizedtypes() 
    
  
    // Begin method overload boilerplate
    template <typename RNG>
    inline void write_array(RNG& base_rng,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,
                            Eigen::Matrix<double,Eigen::Dynamic,1>& vars,
                            const bool emit_transformed_parameters = true,
                            const bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  ((((n_matches + n_matches) + n_matches) + 1) + 1);
      const size_t num_transformed = emit_transformed_parameters * 
  (n_matches + n_matches);
      const size_t num_gen_quantities = emit_generated_quantities * 
  (n_matches + n_matches);
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      std::vector<int> params_i;
      vars = Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <typename RNG>
    inline void write_array(RNG& base_rng, std::vector<double>& params_r,
                            std::vector<int>& params_i,
                            std::vector<double>& vars,
                            bool emit_transformed_parameters = true,
                            bool emit_generated_quantities = true,
                            std::ostream* pstream = nullptr) const {
      const size_t num_params__ = 
  ((((n_matches + n_matches) + n_matches) + 1) + 1);
      const size_t num_transformed = emit_transformed_parameters * 
  (n_matches + n_matches);
      const size_t num_gen_quantities = emit_generated_quantities * 
  (n_matches + n_matches);
      const size_t num_to_write = num_params__ + num_transformed +
        num_gen_quantities;
      vars = std::vector<double>(num_to_write,
        std::numeric_limits<double>::quiet_NaN());
      write_array_impl(base_rng, params_r, params_i, vars,
        emit_transformed_parameters, emit_generated_quantities, pstream);
    }

    template <bool propto__, bool jacobian__, typename T_>
    inline T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r,
                       std::ostream* pstream = nullptr) const {
      Eigen::Matrix<int, -1, 1> params_i;
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }

    template <bool propto__, bool jacobian__, typename T__>
    inline T__ log_prob(std::vector<T__>& params_r,
                        std::vector<int>& params_i,
                        std::ostream* pstream = nullptr) const {
      return log_prob_impl<propto__, jacobian__>(params_r, params_i, pstream);
    }


    inline void transform_inits(const stan::io::var_context& context,
                         Eigen::Matrix<double, Eigen::Dynamic, 1>& params_r,
                         std::ostream* pstream = nullptr) const final {
      std::vector<double> params_r_vec(params_r.size());
      std::vector<int> params_i;
      transform_inits(context, params_i, params_r_vec, pstream);
      params_r = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,1>>(
        params_r_vec.data(), params_r_vec.size());
    }

  inline void transform_inits(const stan::io::var_context& context,
                              std::vector<int>& params_i,
                              std::vector<double>& vars,
                              std::ostream* pstream__ = nullptr) const {
     constexpr std::array<const char*, 5> names__{"baseline", "skill_home",
      "skill_away", "mu_teams", "sigma_teams"};
      const std::array<Eigen::Index, 5> constrain_param_sizes__{n_matches,
       n_matches, n_matches, 1, 1};
      const auto num_constrained_params__ = std::accumulate(
        constrain_param_sizes__.begin(), constrain_param_sizes__.end(), 0);
    
     std::vector<double> params_r_flat__(num_constrained_params__);
     Eigen::Index size_iter__ = 0;
     Eigen::Index flat_iter__ = 0;
     for (auto&& param_name__ : names__) {
       const auto param_vec__ = context.vals_r(param_name__);
       for (Eigen::Index i = 0; i < constrain_param_sizes__[size_iter__]; ++i) {
         params_r_flat__[flat_iter__] = param_vec__[i];
         ++flat_iter__;
       }
       ++size_iter__;
     }
     vars.resize(num_params_r__);
     transform_inits_impl(params_r_flat__, params_i, vars, pstream__);
    } // transform_inits() 
    
};
}
using stan_model = iter2_jera_model_namespace::iter2_jera_model;

#ifndef USING_R

// Boilerplate
stan::model::model_base& new_model(
        stan::io::var_context& data_context,
        unsigned int seed,
        std::ostream* msg_stream) {
  stan_model* m = new stan_model(data_context, seed, msg_stream);
  return *m;
}

stan::math::profile_map& get_stan_profile_data() {
  return iter2_jera_model_namespace::profiles__;
}

#endif


