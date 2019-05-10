/*eslint-disable */
export default [
  {
    path: '/fr',
    component: require('../components/fr/FrEntry.vue').default,
    children: [{
      path: '',
      component: require('../components/fr/FrIndex.vue').default,
      children: [{
        path: '',
        component: require('../pages/fr/FrIntroduction.md').default,
        }
      ],
    },
    {
      path:'theorical',
      component: require('../components/fr/FrComponents.vue').default,
      children: [{
        path: '1',
        component: require('../pages/fr/theoricalRL/Fr1DDPG.md').default,
      }],
    }
    ],
  },
];
/*eslint-disable */
