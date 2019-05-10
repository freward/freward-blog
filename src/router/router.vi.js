/*eslint-disable */
export default [
  {
    path: '/vi',
    component: require('../components/vi/ViEntry.vue').default,
    children: [{
      path: '',
      component: require('../components/vi/ViIndex.vue').default,
      children: [{
        path: '',
        component: require('../pages/vi/ViIntroduction.md').default,
        }
      ],
    },
    {
      path:'theorical',
      component: require('../components/vi/ViComponents.vue').default,
      children: [{
        path: '1',
        component: require('../pages/vi/theoricalRL/Vi1DDPG.md').default,
      }],
    }
    ],
  },
];
/*eslint-disable */
