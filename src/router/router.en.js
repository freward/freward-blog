/*eslint-disable */
export default [
  {
    path: '/',
    component: require('../components/en/EnEntry.vue').default,
    children: [{
      path: '',
      component: require('../components/en/EnIndex.vue').default,
      children: [{
        path: '',
        component: require('../pages/en/EnIntroduction.md').default,
        },
        {
        path: 'about',
        component: require('../components/en/EnAbout.vue').default,
        }
      ],
    },
    {
      path:'theorical',
      component: require('../components/en/EnComponents.vue').default,
      children: [{
        path: '1',
        component: require('../pages/en/theoricalRL/En1DDPG.md').default,
      }],
    }
    ],
  },
];
/*eslint-disable */
