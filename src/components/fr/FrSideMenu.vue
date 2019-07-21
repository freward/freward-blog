<template>
  <aside class="list-group">
    <ul>
      <li><router-link to="/fr" class="list-group-item" exact>Introduction</router-link></li>
      <li><router-link to="/fr/about" class="list-group-item" exact>À propos</router-link></li>
    </ul>
    <div class="menu">
      <p class="menu-label">RL Théorique</p>
      <ul class="menu-list">
        <li><router-link to="/fr/theorical/1" exact>DDPG</router-link></li>
        <li><router-link to="/en/components/layout" exact>TRPO/PPO</router-link></li>
        <li><router-link to="/en/components/button" exact>SAC</router-link></li>
        <li><router-link to="/en/components/icon" exact>Inference</router-link></li>
      </ul>

      <p class="menu-label">RL Practique</p>
      <ul class="menu-list">
        <li><router-link to="/en/components/affix" exact>OpenAI Gym</router-link></li>
        <li><router-link to="/en/components/scrollto" exact>Q-learning</router-link></li>
        <li><router-link to="/en/components/breadcrumb" exact>DDPG</router-link></li>
        <li><router-link to="/en/components/breadcrumb" exact>TRPO/PPO</router-link></li>
        <li><router-link to="/en/components/breadcrumb" exact>SAC</router-link></li>
      </ul>

      <p class="menu-label">Révision bibliothèque </p>
      <ul class="menu-list">
        <li><router-link to="/en/components/alert" exact>OpenAI baselines</router-link></li>
        <li><router-link to="/en/components/aside" exact>Stable baselines</router-link></li>
        <li><router-link to="/en/components/collapse" exact>Ray/RLLib</router-link></li>
        <li><router-link to="/en/components/modal" exact>Unity ML-Agents</router-link></li>
      </ul>

      <p class="menu-label">Application</p>
      <ul class="menu-list">
        <li><router-link to="/en/components/form" exact>Frew<i class="fas fa-angle-double-up" ></i>rd Env</router-link></li>
        <li><router-link to="/en/components/form2" exact>Frew<i class="fas fa-angle-double-up" ></i>rd Exp</router-link></li>
      </ul>
    </div>
  </aside>
</template>

<style scoped>
  .menu-list
  {
  font-size: 14px;
  color: black;
  }
</style>
