<script setup lang="ts">
import { useMereDocsThemeConfig } from '../config'

const themeConfig = useMereDocsThemeConfig()
</script>

<template>
  <section class="mere-atlas-map" aria-label="Mere docs map">
    <div class="mere-atlas-frame">
      <div class="mere-atlas-topline">
        <span>{{ themeConfig.atlas.eyebrowLeft }}</span>
        <span>{{ themeConfig.atlas.eyebrowRight }}</span>
      </div>

      <div class="mere-atlas-core">
        <span class="mere-atlas-core-label">{{ themeConfig.atlas.corePrefix }}</span>
        <span class="mere-atlas-core-dot" aria-hidden="true">.</span>
        <span class="mere-atlas-core-label">{{ themeConfig.atlas.coreSuffix }}</span>
      </div>

      <div class="mere-atlas-planes">
        <a
          v-for="plane in themeConfig.atlas.planes"
          :key="plane.name"
          class="mere-atlas-plane"
          :class="`is-${plane.accent}`"
          :href="plane.href"
        >
          <span class="mere-atlas-plane-name">{{ plane.name }}</span>
          <span class="mere-atlas-plane-signal">{{ plane.signal }}</span>
          <span class="mere-atlas-plane-items">
            <span v-for="item in plane.items" :key="item">{{ item }}</span>
          </span>
        </a>
      </div>
    </div>
  </section>
</template>
