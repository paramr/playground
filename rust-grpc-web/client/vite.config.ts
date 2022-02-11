import { defineConfig } from 'vite'
import { svelte } from '@sveltejs/vite-plugin-svelte'
import sveltePreprocess from 'svelte-preprocess';
import wasmPack from 'vite-plugin-wasm-pack';

const production = process.env.NODE_ENV === 'production'

// https://vitejs.dev/config/
export default defineConfig({
  clearScreen: false,
  plugins: [
    svelte({
			preprocess: [sveltePreprocess({ typescript: true })]
    }),
    wasmPack('./wasm')
  ],
})
