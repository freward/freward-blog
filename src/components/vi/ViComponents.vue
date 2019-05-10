<template>
  <div>
    <nav class="navbar is-fixed-top is-light is-bold">
        <div class="navbar-brand">
            <a class="navbar-item" href="">
                <span class="logo">
                    Frew<i class="fas fa-angle-double-up" ></i>rd
                </span>
            </a>
            <div class="navbar-burger burger" data-target="navbarLanguage">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        <div id="navbarLanguage" class="navbar-menu">
            <div class="navbar-start">
                <div class="navbar-item has-dropdown is-hoverable">
                    <a class="navbar-link">
                        Ngôn Ngữ
                    </a>
                    <div class="navbar-dropdown is-boxed">
                        <a class="navbar-item">
                            <router-link to="/"><span>Tiếng Anh</span></router-link>
                        </a>
                        <a class="navbar-item">
                            <router-link to="/vi"><span>Tiếng Việt</span></router-link>
                        </a>
                        <a class="navbar-item">
                            <router-link to="/fr"><span>Tiếng Pháp</span></router-link>
                        </a>
                    </div>
                </div>
            </div>
            <div class="navbar-end">
                <div class="navbar-item">
                    <div class="field is-grouped">
                        <p class="control">
                            <a class="bd-tw-button button" href="https://github.com/toanngosy/">
                                <span class="icon"><i class="fab fa-github"></i></span>
                            </a>
                        </p>
                        <p class="control">
                            <a class="bd-tw-button button" href="https://www.linkedin.com/in/sy-toan-ngo-491109bb/">
                                <span class="icon"><i class="fab fa-linkedin"></i></span>
                            </a>
                        </p>
                        <p class="control">
                            <a class="bd-tw-button button" href="https://www.facebook.com/toanngosy/">
                                <span class="icon"><i class="fab fa-facebook"></i></span>
                            </a>
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </nav>
    <section class="section">
      <div class="container">
        <div class="columns">
          <div class="column is-2 side-nav">
            <side-menu></side-menu>
          </div>
          <div class="column is-10 markdown">
            <router-view></router-view>
          </div>
        </div>
      </div>
    </section>

    <footer class="footer">
      <div class="container">
        <div class="content has-text-centered">
          <p>
            <strong>Frew<i class="fas fa-angle-double-up" ></i>rd</strong> by <a href="https://www.facebook.com/toanngosy/" target="_blank">Gonaton</a>. Mã nguồn bản quyền
            <a href="http://opensource.org/licenses/mit-license.php">MIT</a>.
          </p>
          <p>
            <a class="icon" href="https://github.com/toanngosy/">
              <i class="fab fa-github"></i>
            </a>
          </p>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import SideMenu from './ViSideMenu';
export default {
  components: { SideMenu }
};
</script>
<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
  .logo
  {
  font-size: 30px;
  color: black;
  }
  .randomimage
  {
  display:table;
  margin:0 auto;
  font-size: 200px;
  color: #D3D3D3;
  }

  .welcome
  {
  display:table;
  margin:0 auto;
  font-size: 62.5px;
  font-weight: bold;
  color: #D3D3D3;
  }
</style>
