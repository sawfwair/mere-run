<script setup lang="ts">
import { useMereDocsThemeConfig } from '../config'

const themeConfig = useMereDocsThemeConfig()
</script>

<template>
  <nav class="mere-docs-network" aria-label="Mere docs network">
    <span class="mere-docs-network-kicker">{{ themeConfig.docsNetworkLabel }}</span>
    <a v-for="link in themeConfig.docsNetworkLinks" :key="`${link.text}:${link.href}`" :href="link.href">
      {{ link.text }}
    </a>
  </nav>
</template>
