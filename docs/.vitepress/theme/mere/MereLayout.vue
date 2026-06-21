<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import MereAtlasMap from './components/MereAtlasMap.vue'
import MereDocSignal from './components/MereDocSignal.vue'
import MereDocsNetwork from './components/MereDocsNetwork.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #nav-bar-title-before>
      <span class="mere-nav-mark" aria-hidden="true" />
    </template>
    <template #home-hero-image>
      <MereAtlasMap />
    </template>
    <template #doc-before>
      <MereDocSignal />
    </template>
    <template #aside-outline-after>
      <MereDocsNetwork />
    </template>
  </Layout>
</template>
