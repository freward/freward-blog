<template>
  <table class="table is-fluid">
    <tbody>
      <tr>
        <td><figure class="image is-256x256">
              <img class="is-rounded" src="https://bulma.io/images/placeholders/128x128.png">
            </figure>
        </td>
        <td><figure class="image is-256x256">
              <img class="is-rounded" src="https://bulma.io/images/placeholders/128x128.png">
            </figure>
        </td>
        <td><figure class="image is-256x256">
              <img class="is-rounded" src="https://bulma.io/images/placeholders/128x128.png">
            </figure>
        </td>
        <td><figure class="image is-256x256">
              <img class="is-rounded" src="https://bulma.io/images/placeholders/128x128.png">
            </figure>
        </td>
      </tr>
      <tr>
        <td>
          template
        </td>
        <td>
          template
        </td>
        <td>
          template
        </td>
        <td>
          template
        </td>
      </tr>
    </tbody>
  </table>
</template>

<!---------------------------------------------------------------->
<script>
import SideMenu from './EnSideMenu'
export default {
  components: { SideMenu },
  methods: {
    reRender () {
      if (window.MathJax) {
        window.MathJax.Hub.Queue(['Typeset', window.MathJax.Hub], () => console.log('done'))
      }
    }
  },
  mounted () {
    this.reRender()
  },
  watch: {
    latex: function () {
      this.$nextTick().then(() => {
        this.reRender()
      })
    }
  }
}
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
