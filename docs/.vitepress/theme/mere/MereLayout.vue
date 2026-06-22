<script setup lang="ts">
import { computed } from 'vue'
import DefaultTheme from 'vitepress/theme'
import MereAtlasMap from './components/MereAtlasMap.vue'
import MereDocSignal from './components/MereDocSignal.vue'
import MereDocsNetwork from './components/MereDocsNetwork.vue'
import { useMereDocsThemeConfig } from './config'

const { Layout } = DefaultTheme
const themeConfig = useMereDocsThemeConfig()

const keyColorStyle = computed(() => ({
  '--mere-key': themeConfig.keyColor.light,
  '--mere-key-hover': themeConfig.keyColor.lightHover,
  '--mere-key-contrast': themeConfig.keyColor.lightContrast,
  '--mere-key-dark': themeConfig.keyColor.dark,
  '--mere-key-dark-hover': themeConfig.keyColor.darkHover,
  '--mere-key-dark-contrast': themeConfig.keyColor.darkContrast,
}))
</script>

<template>
  <div class="mere-docs-theme" :style="keyColorStyle">
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
  </div>
</template>
