/* Frames per second of the Genesis Plus GX libretro core on one ROM,
 * headless (video and audio callbacks do nothing), one process.
 *
 *   gcc -O2 -o /tmp/gpgx_bench inst/examples/gpgx_bench.c -ldl
 *   unzip -p "/mnt/Data2/ROMs/Mortal Kombat 3 (USA).zip" '*.gen' > /tmp/mk3.gen
 *   /tmp/gpgx_bench /usr/lib/x86_64-linux-gnu/libretro/genesis_plus_gx_libretro.so /tmp/mk3.gen [frames=20000]
 *
 * Prints: frames/s of retro_run, time of retro_serialize / retro_unserialize
 * (a reset from a snapshot), and RAM bytes at the MK3 HP / timer addresses
 * (plain and with the byte address XOR 1, to see the core's byte order).
 * Only the libretro declarations used here, copied from libretro.h. */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct retro_game_info { const char *path; const void *data; size_t size; const char *meta; };
typedef bool (*env_t)(unsigned, void *);
typedef void (*video_t)(const void *, unsigned, unsigned, size_t);
typedef void (*audio_t)(int16_t, int16_t);
typedef size_t (*audio_batch_t)(const int16_t *, size_t);
typedef void (*poll_t)(void);
typedef int16_t (*state_t)(unsigned, unsigned, unsigned, unsigned);

#define ENV_GET_CAN_DUPE 3
#define ENV_GET_SYSTEM_DIRECTORY 9
#define ENV_SET_PIXEL_FORMAT 10
#define ENV_GET_SAVE_DIRECTORY 31
#define MEMORY_SYSTEM_RAM 2

static bool env(unsigned cmd, void *data) {
  switch (cmd) {
  case ENV_GET_CAN_DUPE: *(bool *)data = true; return true;
  case ENV_GET_SYSTEM_DIRECTORY:
  case ENV_GET_SAVE_DIRECTORY: *(const char **)data = "/tmp"; return true;
  case ENV_SET_PIXEL_FORMAT: return true;
  default: return false;
  }
}
static void video(const void *d, unsigned w, unsigned h, size_t p) { (void)d; (void)w; (void)h; (void)p; }
static void audio(int16_t l, int16_t r) { (void)l; (void)r; }
static size_t audio_batch(const int16_t *d, size_t n) { (void)d; return n; }
static void poll(void) {}
static int16_t input(unsigned p, unsigned d, unsigned i, unsigned id) { (void)p; (void)d; (void)i; (void)id; return 0; }

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

#define SYM(T, name) T name = (T)dlsym(h, #name); if (!name) { fprintf(stderr, "no %s\n", #name); return 1; }

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s core.so rom.gen [frames]\n", argv[0]); return 1; }
  long frames = argc > 3 ? atol(argv[3]) : 20000;
  void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
  typedef void (*set_env_f)(env_t);
  typedef void (*set_video_f)(video_t);
  typedef void (*set_audio_f)(audio_t);
  typedef void (*set_audio_batch_f)(audio_batch_t);
  typedef void (*set_poll_f)(poll_t);
  typedef void (*set_state_f)(state_t);
  typedef void (*void_f)(void);
  typedef bool (*load_f)(const struct retro_game_info *);
  typedef size_t (*size_f)(void);
  typedef bool (*ser_f)(void *, size_t);
  typedef bool (*unser_f)(const void *, size_t);
  typedef void *(*mem_f)(unsigned);
  typedef size_t (*mem_size_f)(unsigned);
  SYM(set_env_f, retro_set_environment)
  SYM(set_video_f, retro_set_video_refresh)
  SYM(set_audio_f, retro_set_audio_sample)
  SYM(set_audio_batch_f, retro_set_audio_sample_batch)
  SYM(set_poll_f, retro_set_input_poll)
  SYM(set_state_f, retro_set_input_state)
  SYM(void_f, retro_init)
  SYM(load_f, retro_load_game)
  SYM(void_f, retro_run)
  SYM(size_f, retro_serialize_size)
  SYM(ser_f, retro_serialize)
  SYM(unser_f, retro_unserialize)
  SYM(mem_f, retro_get_memory_data)
  SYM(mem_size_f, retro_get_memory_size)

  FILE *f = fopen(argv[2], "rb");
  if (!f) { perror(argv[2]); return 1; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  void *rom = malloc(n);
  if (fread(rom, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
  fclose(f);

  retro_set_environment(env);
  retro_init();
  retro_set_video_refresh(video);
  retro_set_audio_sample(audio);
  retro_set_audio_sample_batch(audio_batch);
  retro_set_input_poll(poll);
  retro_set_input_state(input);
  struct retro_game_info gi = { argv[2], rom, (size_t)n, NULL };
  if (!retro_load_game(&gi)) { fprintf(stderr, "retro_load_game failed\n"); return 1; }

  for (int i = 0; i < 600; i++) retro_run();   /* boot */
  double t0 = now();
  for (long i = 0; i < frames; i++) retro_run();
  double dt = now() - t0;
  printf("%ld frames in %.2f s: %.0f frames/s (MAME: ~440 per process)\n", frames, dt, frames / dt);

  size_t ss = retro_serialize_size();
  void *buf = malloc(ss);
  int reps = 1000;
  t0 = now();
  for (int i = 0; i < reps; i++) retro_serialize(buf, ss);
  double ts = (now() - t0) / reps;
  t0 = now();
  for (int i = 0; i < reps; i++) retro_unserialize(buf, ss);
  double tu = (now() - t0) / reps;
  printf("state %zu bytes: serialize %.3f ms, unserialize %.3f ms\n", ss, 1e3 * ts, 1e3 * tu);

  uint8_t *ram = retro_get_memory_data(MEMORY_SYSTEM_RAM);
  size_t rs = retro_get_memory_size(MEMORY_SYSTEM_RAM);
  printf("system RAM %zu bytes at %p\n", rs, (void *)ram);
  if (ram && rs >= 0x10000) {
    unsigned a[] = { 0xF865, 0xF9D7, 0xF5A3, 0xF5A5 };
    const char *nm[] = { "hp1", "hp2", "t10", "t1" };
    for (int i = 0; i < 4; i++)
      printf("  %s 0xFF%04X: plain %3u, xor1 %3u\n", nm[i], a[i], ram[a[i]], ram[a[i] ^ 1]);
  }
  return 0;
}
