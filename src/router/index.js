import Vue from 'vue'
import Router from 'vue-router'
import EnRouter from './router.en'
import ViRouter from './router.vi'
import FrRouter from './router.fr'
Vue.use(Router)

let router = []
router = router.concat(EnRouter)
router = router.concat(ViRouter)
router = router.concat(FrRouter)

export default new Router({
  routes: router
})
