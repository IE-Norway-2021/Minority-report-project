/* stb_image_write - v1.16 - public domain - http://nothings.org/stb
   writes out PNG/BMP/TGA/JPEG/HDR images to C stdio - Sean Barrett 2010-2015
                                     no warranty implied; use at your own risk
   Before #including,
       #define STB_IMAGE_WRITE_IMPLEMENTATION
   in the file that you want to have the implementation.
   Will probably not work correctly with strict-aliasing optimizations.
ABOUT:
   This header file is a library for writing images to C stdio or a callback.
   The PNG output is not optimal; it is 20-50% larger than the file
   written by a decent optimizing implementation; though providing a custom
   zlib compress function (see STBIW_ZLIB_COMPRESS) can mitigate that.
   This library is designed for source code compactness and simplicity,
   not optimal image file size or run-time performance.
BUILDING:
   You can #define STBIW_ASSERT(x) before the #include to avoid using assert.h.
   You can #define STBIW_MALLOC(), STBIW_REALLOC(), and STBIW_FREE() to replace
   malloc,realloc,free.
   You can #define STBIW_MEMMOVE() to replace memmove()
   You can #define STBIW_ZLIB_COMPRESS to use a custom zlib-style compress function
   for PNG compression (instead of the builtin one), it must have the following signature:
   unsigned char * my_compress(unsigned char *data, int data_len, int *out_len, int quality);
   The returned data will be freed with STBIW_FREE() (free() by default),
   so it must be heap allocated with STBIW_MALLOC() (malloc() by default),
UNICODE:
   If compiling for Windows and you wish to use Unicode filenames, compile
   with
       #define STBIW_WINDOWS_UTF8
   and pass utf8-encoded filenames. Call stbiw_convert_wchar_to_utf8 to convert
   Windows wchar_t filenames to utf8.
USAGE:
   There are five functions, one for each image file format:
     int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
     int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
     int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
     int stbi_write_jpg(char const *filename, int w, int h, int comp, const void *data, int quality);
     int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
     void stbi_flip_vertically_on_write(int flag); // flag is non-zero to flip data vertically
   There are also five equivalent functions that use an arbitrary write function. You are
   expected to open/close your file-equivalent before and after calling these:
     int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data, int stride_in_bytes);
     int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data);
     int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void  *data);
     int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data);
     int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data, int quality);
   where the callback is:
      void stbi_write_func(void *context, void *data, int size);
   You can configure it with these global variables:
      int stbi_write_tga_with_rle;             // defaults to true; set to 0 to disable RLE
      int stbi_write_png_compression_level;    // defaults to 8; set to higher for more compression
      int stbi_write_force_png_filter;         // defaults to -1; set to 0..5 to force a filter mode
   You can define STBI_WRITE_NO_STDIO to disable the file variant of these
   functions, so the library will not use stdio.h at all. However, this will
   also disable HDR writing, because it requires stdio for formatted output.
   Each function returns 0 on failure and non-0 on success.
   The functions create an image file defined by the parameters. The image
   is a rectangle of pixels stored from left-to-right, top-to-bottom.
   Each pixel contains 'comp' channels of data stored interleaved with 8-bits
   per channel, in the following order: 1=Y, 2=YA, 3=RGB, 4=RGBA. (Y is
   monochrome color.) The rectangle is 'w' pixels wide and 'h' pixels tall.
   The *data pointer points to the first byte of the top-left-most pixel.
   For PNG, "stride_in_bytes" is the distance in bytes from the first byte of
   a row of pixels to the first byte of the next row of pixels.
   PNG creates output files with the same number of components as the input.
   The BMP format expands Y to RGB in the file format and does not
   output alpha.
   PNG supports writing rectangles of data even when the bytes storing rows of
   data are not consecutive in memory (e.g. sub-rectangles of a larger image),
   by supplying the stride between the beginning of adjacent rows. The other
   formats do not. (Thus you cannot write a native-format BMP through the BMP
   writer, both because it is in BGR order and because it may have padding
   at the end of the line.)
   PNG allows you to set the deflate compression level by setting the global
   variable 'stbi_write_png_compression_level' (it defaults to 8).
   HDR expects linear float data. Since the format is always 32-bit rgb(e)
   data, alpha (if provided) is discarded, and for monochrome data it is
   replicated across all three channels.
   TGA supports RLE or non-RLE compressed data. To use non-RLE-compressed
   data, set the global variable 'stbi_write_tga_with_rle' to 0.
   JPEG does ignore alpha channels in input data; quality is between 1 and 100.
   Higher quality looks better but results in a bigger image.
   JPEG baseline (no JPEG progressive).
CREDITS:
   Sean Barrett           -    PNG/BMP/TGA
   Baldur Karlsson        -    HDR
   Jean-Sebastien Guay    -    TGA monochrome
   Tim Kelsey             -    misc enhancements
   Alan Hickman           -    TGA RLE
   Emmanuel Julien        -    initial file IO callback implementation
   Jon Olick              -    original jo_jpeg.cpp code
   Daniel Gibson          -    integrate JPEG, allow external zlib
   Aarni Koskela          -    allow choosing PNG filter
   bugfixes:
      github:Chribba
      Guillaume Chereau
      github:jry2
      github:romigrou
      Sergio Gonzalez
      Jonas Karlsson
      Filip Wasil
      Thatcher Ulrich
      github:poppolopoppo
      Patrick Boettcher
      github:xeekworx
      Cap Petschulat
      Simon Rodriguez
      Ivan Tikhonov
      github:ignotion
      Adam Schackart
      Andrew Kensler
LICENSE
  See end of file for license information.
*/

#ifndef INCLUDE_STB_IMAGE_WRITE_H
#define INCLUDE_STB_IMAGE_WRITE_H

#include <stdlib.h>

// if STB_IMAGE_WRITE_STATIC causes problems, try defining STBIWDEF to 'inline' or 'static inline'
#ifndef STBIWDEF
#ifdef STB_IMAGE_WRITE_STATIC
#define STBIWDEF static
#else
#ifdef __cplusplus
#define STBIWDEF extern "C"
#else
#define STBIWDEF extern
#endif
#endif
#endif

#ifndef STB_IMAGE_WRITE_STATIC // C++ forbids static forward declarations
STBIWDEF int stbi_write_tga_with_rle;
STBIWDEF int stbi_write_png_compression_level;
STBIWDEF int stbi_write_force_png_filter;
#endif

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_png(char const *filename, int w, int h, int comp, const void *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp(char const *filename, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_tga(char const *filename, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_hdr(char const *filename, int w, int h, int comp, const float *data);
STBIWDEF int stbi_write_jpg(char const *filename, int x, int y, int comp, const void *data, int quality);

#ifdef STBIW_WINDOWS_UTF8
STBIWDEF int stbiw_convert_wchar_to_utf8(char *buffer, size_t bufferlen, const wchar_t *input);
#endif
#endif

typedef void stbi_write_func(void *context, void *data, int size);

STBIWDEF int stbi_write_png_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data, int stride_in_bytes);
STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const void *data);
STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int w, int h, int comp, const float *data);
STBIWDEF int stbi_write_jpg_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data, int quality);

STBIWDEF void stbi_flip_vertically_on_write(int flip_boolean);

#endif // INCLUDE_STB_IMAGE_WRITE_H

#ifdef STB_IMAGE_WRITE_IMPLEMENTATION

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifndef _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_NONSTDC_NO_DEPRECATE
#endif
#endif

#ifndef STBI_WRITE_NO_STDIO
#include <stdio.h>
#endif // STBI_WRITE_NO_STDIO

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#if defined(STBIW_MALLOC) && defined(STBIW_FREE) && (defined(STBIW_REALLOC) || defined(STBIW_REALLOC_SIZED))
// ok
#elif !defined(STBIW_MALLOC) && !defined(STBIW_FREE) && !defined(STBIW_REALLOC) && !defined(STBIW_REALLOC_SIZED)
// ok
#else
#error "Must define all or none of STBIW_MALLOC, STBIW_FREE, and STBIW_REALLOC (or STBIW_REALLOC_SIZED)."
#endif

#ifndef STBIW_MALLOC
#define STBIW_MALLOC(sz) malloc(sz)
#define STBIW_REALLOC(p, newsz) realloc(p, newsz)
#define STBIW_FREE(p) free(p)
#endif

#ifndef STBIW_REALLOC_SIZED
#define STBIW_REALLOC_SIZED(p, oldsz, newsz) STBIW_REALLOC(p, newsz)
#endif

#ifndef STBIW_MEMMOVE
#define STBIW_MEMMOVE(a, b, sz) memmove(a, b, sz)
#endif

#ifndef STBIW_ASSERT
#include <assert.h>
#define STBIW_ASSERT(x) assert(x)
#endif

#define STBIW_UCHAR(x) (unsigned char)((x)&0xff)

#ifdef STB_IMAGE_WRITE_STATIC
static int stbi_write_png_compression_level = 8;
static int stbi_write_tga_with_rle = 1;
static int stbi_write_force_png_filter = -1;
#else
int stbi_write_png_compression_level = 8;
int stbi_write_tga_with_rle = 1;
int stbi_write_force_png_filter = -1;
#endif

static int stbi__flip_vertically_on_write = 0;

STBIWDEF void stbi_flip_vertically_on_write(int flag) { stbi__flip_vertically_on_write = flag; }

typedef struct {
   stbi_write_func *func;
   void *context;
   unsigned char buffer[64];
   int buf_used;
} stbi__write_context;

// initialize a callback-based context
static void stbi__start_write_callbacks(stbi__write_context *s, stbi_write_func *c, void *context) {
   s->func = c;
   s->context = context;
}

#ifndef STBI_WRITE_NO_STDIO

static void stbi__stdio_write(void *context, void *data, int size) { fwrite(data, 1, size, (FILE *)context); }

#if defined(_WIN32) && defined(STBIW_WINDOWS_UTF8)
#ifdef __cplusplus
#define STBIW_EXTERN extern "C"
#else
#define STBIW_EXTERN extern
#endif
STBIW_EXTERN __declspec(dllimport) int __stdcall MultiByteToWideChar(unsigned int cp, unsigned long flags, const char *str, int cbmb, wchar_t *widestr,
                                                                     int cchwide);
STBIW_EXTERN __declspec(dllimport) int __stdcall WideCharToMultiByte(unsigned int cp, unsigned long flags, const wchar_t *widestr, int cchwide, char *str,
                                                                     int cbmb, const char *defchar, int *used_default);

STBIWDEF int stbiw_convert_wchar_to_utf8(char *buffer, size_t bufferlen, const wchar_t *input) {
   return WideCharToMultiByte(65001 /* UTF8 */, 0, input, -1, buffer, (int)bufferlen, NULL, NULL);
}
#endif

static FILE *stbiw__fopen(char const *filename, char const *mode) {
   FILE *f;
#if defined(_WIN32) && defined(STBIW_WINDOWS_UTF8)
   wchar_t wMode[64];
   wchar_t wFilename[1024];
   if (0 == MultiByteToWideChar(65001 /* UTF8 */, 0, filename, -1, wFilename, sizeof(wFilename) / sizeof(*wFilename)))
      return 0;

   if (0 == MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, wMode, sizeof(wMode) / sizeof(*wMode)))
      return 0;

#if defined(_MSC_VER) && _MSC_VER >= 1400
   if (0 != _wfopen_s(&f, wFilename, wMode))
      f = 0;
#else
   f = _wfopen(wFilename, wMode);
#endif

#elif defined(_MSC_VER) && _MSC_VER >= 1400
   if (0 != fopen_s(&f, filename, mode))
      f = 0;
#else
   f = fopen(filename, mode);
#endif
   return f;
}

static int stbi__start_write_file(stbi__write_context *s, const char *filename) {
   FILE *f = stbiw__fopen(filename, "wb");
   stbi__start_write_callbacks(s, stbi__stdio_write, (void *)f);
   return f != NULL;
}

static void stbi__end_write_file(stbi__write_context *s) { fclose((FILE *)s->context); }

#endif // !STBI_WRITE_NO_STDIO

typedef unsigned int stbiw_uint32;
typedef int stb_image_write_test[sizeof(stbiw_uint32) == 4 ? 1 : -1];

static void stbiw__writefv(stbi__write_context *s, const char *fmt, va_list v) {
   while (*fmt) {
      switch (*fmt++) {
      case ' ':
         break;
      case '1': {
         unsigned char x = STBIW_UCHAR(va_arg(v, int));
         s->func(s->context, &x, 1);
         break;
      }
      case '2': {
         int x = va_arg(v, int);
         unsigned char b[2];
         b[0] = STBIW_UCHAR(x);
         b[1] = STBIW_UCHAR(x >> 8);
         s->func(s->context, b, 2);
         break;
      }
      case '4': {
         stbiw_uint32 x = va_arg(v, int);
         unsigned char b[4];
         b[0] = STBIW_UCHAR(x);
         b[1] = STBIW_UCHAR(x >> 8);
         b[2] = STBIW_UCHAR(x >> 16);
         b[3] = STBIW_UCHAR(x >> 24);
         s->func(s->context, b, 4);
         break;
      }
      default:
         STBIW_ASSERT(0);
         return;
      }
   }
}

static void stbiw__writef(stbi__write_context *s, const char *fmt, ...) {
   va_list v;
   va_start(v, fmt);
   stbiw__writefv(s, fmt, v);
   va_end(v);
}

static void stbiw__write_flush(stbi__write_context *s) {
   if (s->buf_used) {
      s->func(s->context, &s->buffer, s->buf_used);
      s->buf_used = 0;
   }
}

static void stbiw__putc(stbi__write_context *s, unsigned char c) { s->func(s->context, &c, 1); }

static void stbiw__write1(stbi__write_context *s, unsigned char a) {
   if ((size_t)s->buf_used + 1 > sizeof(s->buffer))
      stbiw__write_flush(s);
   s->buffer[s->buf_used++] = a;
}

static void stbiw__write3(stbi__write_context *s, unsigned char a, unsigned char b, unsigned char c) {
   int n;
   if ((size_t)s->buf_used + 3 > sizeof(s->buffer))
      stbiw__write_flush(s);
   n = s->buf_used;
   s->buf_used = n + 3;
   s->buffer[n + 0] = a;
   s->buffer[n + 1] = b;
   s->buffer[n + 2] = c;
}

static void stbiw__write_pixel(stbi__write_context *s, int rgb_dir, int comp, int write_alpha, int expand_mono, unsigned char *d) {
   unsigned char bg[3] = {255, 0, 255}, px[3];
   int k;

   if (write_alpha < 0)
      stbiw__write1(s, d[comp - 1]);

   switch (comp) {
   case 2: // 2 pixels = mono + alpha, alpha is written separately, so same as 1-channel case
   case 1:
      if (expand_mono)
         stbiw__write3(s, d[0], d[0], d[0]); // monochrome bmp
      else
         stbiw__write1(s, d[0]); // monochrome TGA
      break;
   case 4:
      if (!write_alpha) {
         // composite against pink background
         for (k = 0; k < 3; ++k)
            px[k] = bg[k] + ((d[k] - bg[k]) * d[3]) / 255;
         stbiw__write3(s, px[1 - rgb_dir], px[1], px[1 + rgb_dir]);
         break;
      }
      /* FALLTHROUGH */
   case 3:
      stbiw__write3(s, d[1 - rgb_dir], d[1], d[1 + rgb_dir]);
      break;
   }
   if (write_alpha > 0)
      stbiw__write1(s, d[comp - 1]);
}

static void stbiw__write_pixels(stbi__write_context *s, int rgb_dir, int vdir, int x, int y, int comp, void *data, int write_alpha, int scanline_pad,
                                int expand_mono) {
   stbiw_uint32 zero = 0;
   int i, j, j_end;

   if (y <= 0)
      return;

   if (stbi__flip_vertically_on_write)
      vdir *= -1;

   if (vdir < 0) {
      j_end = -1;
      j = y - 1;
   } else {
      j_end = y;
      j = 0;
   }

   for (; j != j_end; j += vdir) {
      for (i = 0; i < x; ++i) {
         unsigned char *d = (unsigned char *)data + (j * x + i) * comp;
         stbiw__write_pixel(s, rgb_dir, comp, write_alpha, expand_mono, d);
      }
      stbiw__write_flush(s);
      s->func(s->context, &zero, scanline_pad);
   }
}

static int stbiw__outfile(stbi__write_context *s, int rgb_dir, int vdir, int x, int y, int comp, int expand_mono, void *data, int alpha, int pad,
                          const char *fmt, ...) {
   if (y < 0 || x < 0) {
      return 0;
   } else {
      va_list v;
      va_start(v, fmt);
      stbiw__writefv(s, fmt, v);
      va_end(v);
      stbiw__write_pixels(s, rgb_dir, vdir, x, y, comp, data, alpha, pad, expand_mono);
      return 1;
   }
}

static int stbi_write_bmp_core(stbi__write_context *s, int x, int y, int comp, const void *data) {
   if (comp != 4) {
      // write RGB bitmap
      int pad = (-x * 3) & 3;
      return stbiw__outfile(s, -1, -1, x, y, comp, 1, (void *)data, 0, pad,
                            "11 4 22 4"
                            "4 44 22 444444",
                            'B', 'M', 14 + 40 + (x * 3 + pad) * y, 0, 0, 14 + 40, // file header
                            40, x, y, 1, 24, 0, 0, 0, 0, 0, 0);                   // bitmap header
   } else {
      // RGBA bitmaps need a v4 header
      // use BI_BITFIELDS mode with 32bpp and alpha mask
      // (straight BI_RGB with alpha mask doesn't work in most readers)
      return stbiw__outfile(s, -1, -1, x, y, comp, 1, (void *)data, 1, 0,
                            "11 4 22 4"
                            "4 44 22 444444 4444 4 444 444 444 444",
                            'B', 'M', 14 + 108 + x * y * 4, 0, 0, 14 + 108,                                                                  // file header
                            108, x, y, 1, 32, 3, 0, 0, 0, 0, 0, 0xff0000, 0xff00, 0xff, 0xff000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // bitmap V4 header
   }
}

STBIWDEF int stbi_write_bmp_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data) {
   stbi__write_context s = {0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_bmp_core(&s, x, y, comp, data);
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_bmp(char const *filename, int x, int y, int comp, const void *data) {
   stbi__write_context s = {0};
   if (stbi__start_write_file(&s, filename)) {
      int r = stbi_write_bmp_core(&s, x, y, comp, data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif //! STBI_WRITE_NO_STDIO

static int stbi_write_tga_core(stbi__write_context *s, int x, int y, int comp, void *data) {
   int has_alpha = (comp == 2 || comp == 4);
   int colorbytes = has_alpha ? comp - 1 : comp;
   int format = colorbytes < 2 ? 3 : 2; // 3 color channels (RGB/RGBA) = 2, 1 color channel (Y/YA) = 3

   if (y < 0 || x < 0)
      return 0;

   if (!stbi_write_tga_with_rle) {
      return stbiw__outfile(s, -1, -1, x, y, comp, 0, (void *)data, has_alpha, 0, "111 221 2222 11", 0, 0, format, 0, 0, 0, 0, 0, x, y,
                            (colorbytes + has_alpha) * 8, has_alpha * 8);
   } else {
      int i, j, k;
      int jend, jdir;

      stbiw__writef(s, "111 221 2222 11", 0, 0, format + 8, 0, 0, 0, 0, 0, x, y, (colorbytes + has_alpha) * 8, has_alpha * 8);

      if (stbi__flip_vertically_on_write) {
         j = 0;
         jend = y;
         jdir = 1;
      } else {
         j = y - 1;
         jend = -1;
         jdir = -1;
      }
      for (; j != jend; j += jdir) {
         unsigned char *row = (unsigned char *)data + j * x * comp;
         int len;

         for (i = 0; i < x; i += len) {
            unsigned char *begin = row + i * comp;
            int diff = 1;
            len = 1;

            if (i < x - 1) {
               ++len;
               diff = memcmp(begin, row + (i + 1) * comp, comp);
               if (diff) {
                  const unsigned char *prev = begin;
                  for (k = i + 2; k < x && len < 128; ++k) {
                     if (memcmp(prev, row + k * comp, comp)) {
                        prev += comp;
                        ++len;
                     } else {
                        --len;
                        break;
                     }
                  }
               } else {
                  for (k = i + 2; k < x && len < 128; ++k) {
                     if (!memcmp(begin, row + k * comp, comp)) {
                        ++len;
                     } else {
                        break;
                     }
                  }
               }
            }

            if (diff) {
               unsigned char header = STBIW_UCHAR(len - 1);
               stbiw__write1(s, header);
               for (k = 0; k < len; ++k) {
                  stbiw__write_pixel(s, -1, comp, has_alpha, 0, begin + k * comp);
               }
            } else {
               unsigned char header = STBIW_UCHAR(len - 129);
               stbiw__write1(s, header);
               stbiw__write_pixel(s, -1, comp, has_alpha, 0, begin);
            }
         }
      }
      stbiw__write_flush(s);
   }
   return 1;
}

STBIWDEF int stbi_write_tga_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const void *data) {
   stbi__write_context s = {0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_tga_core(&s, x, y, comp, (void *)data);
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_tga(char const *filename, int x, int y, int comp, const void *data) {
   stbi__write_context s = {0};
   if (stbi__start_write_file(&s, filename)) {
      int r = stbi_write_tga_core(&s, x, y, comp, (void *)data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif

// *************************************************************************************************
// Radiance RGBE HDR writer
// by Baldur Karlsson

#define stbiw__max(a, b) ((a) > (b) ? (a) : (b))

#ifndef STBI_WRITE_NO_STDIO

static void stbiw__linear_to_rgbe(unsigned char *rgbe, float *linear) {
   int exponent;
   float maxcomp = stbiw__max(linear[0], stbiw__max(linear[1], linear[2]));

   if (maxcomp < 1e-32f) {
      rgbe[0] = rgbe[1] = rgbe[2] = rgbe[3] = 0;
   } else {
      float normalize = (float)frexp(maxcomp, &exponent) * 256.0f / maxcomp;

      rgbe[0] = (unsigned char)(linear[0] * normalize);
      rgbe[1] = (unsigned char)(linear[1] * normalize);
      rgbe[2] = (unsigned char)(linear[2] * normalize);
      rgbe[3] = (unsigned char)(exponent + 128);
   }
}

static void stbiw__write_run_data(stbi__write_context *s, int length, unsigned char databyte) {
   unsigned char lengthbyte = STBIW_UCHAR(length + 128);
   STBIW_ASSERT(length + 128 <= 255);
   s->func(s->context, &lengthbyte, 1);
   s->func(s->context, &databyte, 1);
}

static void stbiw__write_dump_data(stbi__write_context *s, int length, unsigned char *data) {
   unsigned char lengthbyte = STBIW_UCHAR(length);
   STBIW_ASSERT(length <= 128); // inconsistent with spec but consistent with official code
   s->func(s->context, &lengthbyte, 1);
   s->func(s->context, data, length);
}

static void stbiw__write_hdr_scanline(stbi__write_context *s, int width, int ncomp, unsigned char *scratch, float *scanline) {
   unsigned char scanlineheader[4] = {2, 2, 0, 0};
   unsigned char rgbe[4];
   float linear[3];
   int x;

   scanlineheader[2] = (width & 0xff00) >> 8;
   scanlineheader[3] = (width & 0x00ff);

   /* skip RLE for images too small or large */
   if (width < 8 || width >= 32768) {
      for (x = 0; x < width; x++) {
         switch (ncomp) {
         case 4: /* fallthrough */
         case 3:
            linear[2] = scanline[x * ncomp + 2];
            linear[1] = scanline[x * ncomp + 1];
            linear[0] = scanline[x * ncomp + 0];
            break;
         default:
            linear[0] = linear[1] = linear[2] = scanline[x * ncomp + 0];
            break;
         }
         stbiw__linear_to_rgbe(rgbe, linear);
         s->func(s->context, rgbe, 4);
      }
   } else {
      int c, r;
      /* encode into scratch buffer */
      for (x = 0; x < width; x++) {
         switch (ncomp) {
         case 4: /* fallthrough */
         case 3:
            linear[2] = scanline[x * ncomp + 2];
            linear[1] = scanline[x * ncomp + 1];
            linear[0] = scanline[x * ncomp + 0];
            break;
         default:
            linear[0] = linear[1] = linear[2] = scanline[x * ncomp + 0];
            break;
         }
         stbiw__linear_to_rgbe(rgbe, linear);
         scratch[x + width * 0] = rgbe[0];
         scratch[x + width * 1] = rgbe[1];
         scratch[x + width * 2] = rgbe[2];
         scratch[x + width * 3] = rgbe[3];
      }

      s->func(s->context, scanlineheader, 4);

      /* RLE each component separately */
      for (c = 0; c < 4; c++) {
         unsigned char *comp = &scratch[width * c];

         x = 0;
         while (x < width) {
            // find first run
            r = x;
            while (r + 2 < width) {
               if (comp[r] == comp[r + 1] && comp[r] == comp[r + 2])
                  break;
               ++r;
            }
            if (r + 2 >= width)
               r = width;
            // dump up to first run
            while (x < r) {
               int len = r - x;
               if (len > 128)
                  len = 128;
               stbiw__write_dump_data(s, len, &comp[x]);
               x += len;
            }
            // if there's a run, output it
            if (r + 2 < width) { // same test as what we break out of in search loop, so only true if we break'd
               // find next byte after run
               while (r < width && comp[r] == comp[x])
                  ++r;
               // output run up to r
               while (x < r) {
                  int len = r - x;
                  if (len > 127)
                     len = 127;
                  stbiw__write_run_data(s, len, comp[x]);
                  x += len;
               }
            }
         }
      }
   }
}

static int stbi_write_hdr_core(stbi__write_context *s, int x, int y, int comp, float *data) {
   if (y <= 0 || x <= 0 || data == NULL)
      return 0;
   else {
      // Each component is stored separately. Allocate scratch space for full output scanline.
      unsigned char *scratch = (unsigned char *)STBIW_MALLOC(x * 4);
      int i, len;
      char buffer[128];
      char header[] = "#?RADIANCE\n# Written by stb_image_write.h\nFORMAT=32-bit_rle_rgbe\n";
      s->func(s->context, header, sizeof(header) - 1);

#ifdef __STDC_LIB_EXT1__
      len = sprintf_s(buffer, sizeof(buffer), "EXPOSURE=          1.0000000000000\n\n-Y %d +X %d\n", y, x);
#else
      len = sprintf(buffer, "EXPOSURE=          1.0000000000000\n\n-Y %d +X %d\n", y, x);
#endif
      s->func(s->context, buffer, len);

      for (i = 0; i < y; i++)
         stbiw__write_hdr_scanline(s, x, comp, scratch, data + comp * x * (stbi__flip_vertically_on_write ? y - 1 - i : i));
      STBIW_FREE(scratch);
      return 1;
   }
}

STBIWDEF int stbi_write_hdr_to_func(stbi_write_func *func, void *context, int x, int y, int comp, const float *data) {
   stbi__write_context s = {0};
   stbi__start_write_callbacks(&s, func, context);
   return stbi_write_hdr_core(&s, x, y, comp, (float *)data);
}

STBIWDEF int stbi_write_hdr(char const *filename, int x, int y, int comp, const float *data) {
   stbi__write_context s = {0};
   if (stbi__start_write_file(&s, filename)) {
      int r = stbi_write_hdr_core(&s, x, y, comp, (float *)data);
      stbi__end_write_file(&s);
      return r;
   } else
      return 0;
}
#endif // STBI_WRITE_NO_STDIO

//////////////////////////////////////////////////////////////////////////////
//
// PNG writer
//

#ifndef STBIW_ZLIB_COMPRESS
// stretchy buffer; stbiw__sbpush() == vector<>::push_back() -- stbiw__sbcount() == vector<>::size()
#define stbiw__sbraw(a) ((int *)(void *)(a)-2)
#define stbiw__sbm(a) stbiw__sbraw(a)[0]
#define stbiw__sbn(a) stbiw__sbraw(a)[1]

#define stbiw__sbneedgrow(a, n) ((a) == 0 || stbiw__sbn(a) + n >= stbiw__sbm(a))
#define stbiw__sbmaybegrow(a, n) (stbiw__sbneedgrow(a, (n)) ? stbiw__sbgrow(a, n) : 0)
#define stbiw__sbgrow(a, n) stbiw__sbgrowf((void **)&(a), (n), sizeof(*(a)))

#define stbiw__sbpush(a, v) (stbiw__sbmaybegrow(a, 1), (a)[stbiw__sbn(a)++] = (v))
#define stbiw__sbcount(a) ((a) ? stbiw__sbn(a) : 0)
#define stbiw__sbfree(a) ((a) ? STBIW_FREE(stbiw__sbraw(a)), 0 : 0)

static void *stbiw__sbgrowf(void **arr, int increment, int itemsize) {
   int m = *arr ? 2 * stbiw__sbm(*arr) + increment : increment + 1;
   void *p = STBIW_REALLOC_SIZED(*arr ? stbiw__sbraw(*arr) : 0, *arr ? (stbiw__sbm(*arr) * itemsize + sizeof(int) * 2) : 0, itemsize * m + sizeof(int) * 2);
   STBIW_ASSERT(p);
   if (p) {
      if (!*arr)
         ((int *)p)[1] = 0;
      *arr = (void *)((int *)p + 2);
      stbiw__sbm(*arr) = m;
   }
   return *arr;
}

static unsigned char *stbiw__zlib_flushf(unsigned char *data, unsigned int *bitbuffer, int *bitcount) {
   while (*bitcount >= 8) {
      stbiw__sbpush(data, STBIW_UCHAR(*bitbuffer));
      *bitbuffer >>= 8;
      *bitcount -= 8;
   }
   return data;
}

static int stbiw__zlib_bitrev(int code, int codebits) {
   int res = 0;
   while (codebits--) {
      res = (res << 1) | (code & 1);
      code >>= 1;
   }
   return res;
}

static unsigned int stbiw__zlib_countm(unsigned char *a, unsigned char *b, int limit) {
   int i;
   for (i = 0; i < limit && i < 258; ++i)
      if (a[i] != b[i])
         break;
   return i;
}

static unsigned int stbiw__zhash(unsigned char *data) {
   stbiw_uint32 hash = data[0] + (data[1] << 8) + (data[2] << 16);
   hash ^= hash << 3;
   hash += hash >> 5;
   hash ^= hash << 4;
   hash += hash >> 17;
   hash ^= hash << 25;
   hash += hash >> 6;
   return hash;
}

#define stbiw__zlib_flush() (out = stbiw__zlib_flushf(out, &bitbuf, &bitcount))
#define stbiw__zlib_add(code, codebits) (bitbuf |= (code) << bitcount, bitcount += (codebits), stbiw__zlib_flush())
#define stbiw__zlib_huffa(b, c) stbiw__zlib_add(stbiw__zlib_bitrev(b, c), c)
// default huffman tables
#define stbiw__zlib_huff1(n) stbiw__zlib_huffa(0x30 + (n), 8)
#define stbiw__zlib_huff2(n) stbiw__zlib_huffa(0x190 + (n)-144, 9)
#define stbiw__zlib_huff3(n) stbiw__zlib_huffa(0 + (n)-256, 7)
#define stbiw__zlib_huff4(n) stbiw__zlib_huffa(0xc0 + (n)-280, 8)
#define stbiw__zlib_huff(n) ((n) <= 143 ? stbiw__zlib_huff1(n) : (n) <= 255 ? stbiw__zlib_huff2(n) : (n) <= 279 ? stbiw__zlib_huff3(n) : stbiw__zlib_huff4(n))
#define stbiw__zlib_huffb(n) ((n) <= 143 ? stbiw__zlib_huff1(n) : stbiw__zlib_huff2(n))

#define stbiw__ZHASH 16384

#endif // STBIW_ZLIB_COMPRESS

STBIWDEF unsigned char *stbi_zlib_compress(unsigned char *data, int data_len, int *out_len, int quality) {
#ifdef STBIW_ZLIB_COMPRESS
   // user provided a zlib compress implementation, use that
   return STBIW_ZLIB_COMPRESS(data, data_len, out_len, quality);
#else  // use builtin
   static unsigned short lengthc[] = {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 259};
   static unsigned char lengtheb[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
   static unsigned short distc[] = {1,   2,   3,   4,   5,    7,    9,    13,   17,   25,   33,   49,    65,    97,    129,  193,
                                    257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32768};
   static unsigned char disteb[] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
   unsigned int bitbuf = 0;
   int i, j, bitcount = 0;
   unsigned char *out = NULL;
   unsigned char ***hash_table = (unsigned char ***)STBIW_MALLOC(stbiw__ZHASH * sizeof(unsigned char **));
   if (hash_table == NULL)
      return NULL;
   if (quality < 5)
      quality = 5;

   stbiw__sbpush(out, 0x78); // DEFLATE 32K window
   stbiw__sbpush(out, 0x5e); // FLEVEL = 1
   stbiw__zlib_add(1, 1);    // BFINAL = 1
   stbiw__zlib_add(1, 2);    // BTYPE = 1 -- fixed huffman

   for (i = 0; i < stbiw__ZHASH; ++i)
      hash_table[i] = NULL;

   i = 0;
   while (i < data_len - 3) {
      // hash next 3 bytes of data to be compressed
      int h = stbiw__zhash(data + i) & (stbiw__ZHASH - 1), best = 3;
      unsigned char *bestloc = 0;
      unsigned char **hlist = hash_table[h];
      int n = stbiw__sbcount(hlist);
      for (j = 0; j < n; ++j) {
         if (hlist[j] - data > i - 32768) { // if entry lies within window
            int d = stbiw__zlib_countm(hlist[j], data + i, data_len - i);
            if (d >= best) {
               best = d;
               bestloc = hlist[j];
            }
         }
      }
      // when hash table entry is too long, delete half the entries
      if (hash_table[h] && stbiw__sbn(hash_table[h]) == 2 * quality) {
         STBIW_MEMMOVE(hash_table[h], hash_table[h] + quality, sizeof(hash_table[h][0]) * quality);
         stbiw__sbn(hash_table[h]) = quality;
      }
      stbiw__sbpush(hash_table[h], data + i);

      if (bestloc) {
         // "lazy matching" - check match at *next* byte, and if it's better, do cur byte as literal
         h = stbiw__zhash(data + i + 1) & (stbiw__ZHASH - 1);
         hlist = hash_table[h];
         n = stbiw__sbcount(hlist);
         for (j = 0; j < n; ++j) {
            if (hlist[j] - data > i - 32767) {
               int e = stbiw__zlib_countm(hlist[j], data + i + 1, data_len - i - 1);
               if (e > best) { // if next match is better, bail on current match
                  bestloc = NULL;
                  break;
               }
            }
         }
      }

      if (bestloc) {
         int d = (int)(data + i - bestloc); // distance back
         STBIW_ASSERT(d <= 32767 && best <= 258);
         for (j = 0; best > lengthc[j + 1] - 1; ++j)
            ;
         stbiw__zlib_huff(j + 257);
         if (lengtheb[j])
            stbiw__zlib_add(best - lengthc[j], lengtheb[j]);
         for (j = 0; d > distc[j + 1] - 1; ++j)
            ;
         stbiw__zlib_add(stbiw__zlib_bitrev(j, 5), 5);
         if (disteb[j])
            stbiw__zlib_add(d - distc[j], disteb[j]);
         i += best;
      } else {
         stbiw__zlib_huffb(data[i]);
         ++i;
      }
   }
   // write out final bytes
   for (; i < data_len; ++i)
      stbiw__zlib_huffb(data[i]);
   stbiw__zlib_huff(256); // end of block
   // pad with 0 bits to byte boundary
   while (bitcount)
      stbiw__zlib_add(0, 1);

   for (i = 0; i < stbiw__ZHASH; ++i)
      (void)stbiw__sbfree(hash_table[i]);
   STBIW_FREE(hash_table);

   // store uncompressed instead if compression was worse
   if (stbiw__sbn(out) > data_len + 2 + ((data_len + 32766) / 32767) * 5) {
      stbiw__sbn(out) = 2; // truncate to DEFLATE 32K window and FLEVEL = 1
      for (j = 0; j < data_len;) {
         int blocklen = data_len - j;
         if (blocklen > 32767)
            blocklen = 32767;
         stbiw__sbpush(out, data_len - j == blocklen); // BFINAL = ?, BTYPE = 0 -- no compression
         stbiw__sbpush(out, STBIW_UCHAR(blocklen));    // LEN
         stbiw__sbpush(out, STBIW_UCHAR(blocklen >> 8));
         stbiw__sbpush(out, STBIW_UCHAR(~blocklen)); // NLEN
         stbiw__sbpush(out, STBIW_UCHAR(~blocklen >> 8));
         memcpy(out + stbiw__sbn(out), data + j, blocklen);
         stbiw__sbn(out) += blocklen;
         j += blocklen;
      }
   }

   {
      // compute adler32 on input
      unsigned int s1 = 1, s2 = 0;
      int blocklen = (int)(data_len % 5552);
      j = 0;
      while (j < data_len) {
         for (i = 0; i < blocklen; ++i) {
            s1 += data[j + i];
            s2 += s1;
         }
         s1 %= 65521;
         s2 %= 65521;
         j += blocklen;
         blocklen = 5552;
      }
      stbiw__sbpush(out, STBIW_UCHAR(s2 >> 8));
      stbiw__sbpush(out, STBIW_UCHAR(s2));
      stbiw__sbpush(out, STBIW_UCHAR(s1 >> 8));
      stbiw__sbpush(out, STBIW_UCHAR(s1));
   }
   *out_len = stbiw__sbn(out);
   // make returned pointer freeable
   STBIW_MEMMOVE(stbiw__sbraw(out), out, *out_len);
   return (unsigned char *)stbiw__sbraw(out);
#endif // STBIW_ZLIB_COMPRESS
}

static unsigned int stbiw__crc32(unsigned char *buffer, int len) {
#ifdef STBIW_CRC32
   return STBIW_CRC32(buffer, len);
#else
   static unsigned int crc_table[256] = {
       0x00000000, 0x77073096, 0xEE0E612C, 0x990951BA, 0x076DC419, 0x706AF48F, 0xE963A535, 0x9E6495A3, 0x0eDB8832, 0x79DCB8A4, 0xE0D5E91E, 0x97D2D988,
       0x09B64C2B, 0x7EB17CBD, 0xE7B82D07, 0x90BF1D91, 0x1DB71064, 0x6AB020F2, 0xF3B97148, 0x84BE41DE, 0x1ADAD47D, 0x6DDDE4EB, 0xF4D4B551, 0x83D385C7,
       0x136C9856, 0x646BA8C0, 0xFD62F97A, 0x8A65C9EC, 0x14015C4F, 0x63066CD9, 0xFA0F3D63, 0x8D080DF5, 0x3B6E20C8, 0x4C69105E, 0xD56041E4, 0xA2677172,
       0x3C03E4D1, 0x4B04D447, 0xD20D85FD, 0xA50AB56B, 0x35B5A8FA, 0x42B2986C, 0xDBBBC9D6, 0xACBCF940, 0x32D86CE3, 0x45DF5C75, 0xDCD60DCF, 0xABD13D59,
       0x26D930AC, 0x51DE003A, 0xC8D75180, 0xBFD06116, 0x21B4F4B5, 0x56B3C423, 0xCFBA9599, 0xB8BDA50F, 0x2802B89E, 0x5F058808, 0xC60CD9B2, 0xB10BE924,
       0x2F6F7C87, 0x58684C11, 0xC1611DAB, 0xB6662D3D, 0x76DC4190, 0x01DB7106, 0x98D220BC, 0xEFD5102A, 0x71B18589, 0x06B6B51F, 0x9FBFE4A5, 0xE8B8D433,
       0x7807C9A2, 0x0F00F934, 0x9609A88E, 0xE10E9818, 0x7F6A0DBB, 0x086D3D2D, 0x91646C97, 0xE6635C01, 0x6B6B51F4, 0x1C6C6162, 0x856530D8, 0xF262004E,
       0x6C0695ED, 0x1B01A57B, 0x8208F4C1, 0xF50FC457, 0x65B0D9C6, 0x12B7E950, 0x8BBEB8EA, 0xFCB9887C, 0x62DD1DDF, 0x15DA2D49, 0x8CD37CF3, 0xFBD44C65,
       0x4DB26158, 0x3AB551CE, 0xA3BC0074, 0xD4BB30E2, 0x4ADFA541, 0x3DD895D7, 0xA4D1C46D, 0xD3D6F4FB, 0x4369E96A, 0x346ED9FC, 0xAD678846, 0xDA60B8D0,
       0x44042D73, 0x33031DE5, 0xAA0A4C5F, 0xDD0D7CC9, 0x5005713C, 0x270241AA, 0xBE0B1010, 0xC90C2086, 0x5768B525, 0x206F85B3, 0xB966D409, 0xCE61E49F,
       0x5EDEF90E, 0x29D9C998, 0xB0D09822, 0xC7D7A8B4, 0x59B33D17, 0x2EB40D81, 0xB7BD5C3B, 0xC0BA6CAD, 0xEDB88320, 0x9ABFB3B6, 0x03B6E20C, 0x74B1D29A,
       0xEAD54739, 0x9DD277AF, 0x04DB2615, 0x73DC1683, 0xE3630B12, 0x94643B84, 0x0D6D6A3E, 0x7A6A5AA8, 0xE40ECF0B, 0x9309FF9D, 0x0A00AE27, 0x7D079EB1,
       0xF00F9344, 0x8708A3D2, 0x1E01F268, 0x6906C2FE, 0xF762575D, 0x806567CB, 0x196C3671, 0x6E6B06E7, 0xFED41B76, 0x89D32BE0, 0x10DA7A5A, 0x67DD4ACC,
       0xF9B9DF6F, 0x8EBEEFF9, 0x17B7BE43, 0x60B08ED5, 0xD6D6A3E8, 0xA1D1937E, 0x38D8C2C4, 0x4FDFF252, 0xD1BB67F1, 0xA6BC5767, 0x3FB506DD, 0x48B2364B,
       0xD80D2BDA, 0xAF0A1B4C, 0x36034AF6, 0x41047A60, 0xDF60EFC3, 0xA867DF55, 0x316E8EEF, 0x4669BE79, 0xCB61B38C, 0xBC66831A, 0x256FD2A0, 0x5268E236,
       0xCC0C7795, 0xBB0B4703, 0x220216B9, 0x5505262F, 0xC5BA3BBE, 0xB2BD0B28, 0x2BB45A92, 0x5CB36A04, 0xC2D7FFA7, 0xB5D0CF31, 0x2CD99E8B, 0x5BDEAE1D,
       0x9B64C2B0, 0xEC63F226, 0x756AA39C, 0x026D930A, 0x9C0906A9, 0xEB0E363F, 0x72076785, 0x05005713, 0x95BF4A82, 0xE2B87A14, 0x7BB12BAE, 0x0CB61B38,
       0x92D28E9B, 0xE5D5BE0D, 0x7CDCEFB7, 0x0BDBDF21, 0x86D3D2D4, 0xF1D4E242, 0x68DDB3F8, 0x1FDA836E, 0x81BE16CD, 0xF6B9265B, 0x6FB077E1, 0x18B74777,
       0x88085AE6, 0xFF0F6A70, 0x66063BCA, 0x11010B5C, 0x8F659EFF, 0xF862AE69, 0x616BFFD3, 0x166CCF45, 0xA00AE278, 0xD70DD2EE, 0x4E048354, 0x3903B3C2,
       0xA7672661, 0xD06016F7, 0x4969474D, 0x3E6E77DB, 0xAED16A4A, 0xD9D65ADC, 0x40DF0B66, 0x37D83BF0, 0xA9BCAE53, 0xDEBB9EC5, 0x47B2CF7F, 0x30B5FFE9,
       0xBDBDF21C, 0xCABAC28A, 0x53B39330, 0x24B4A3A6, 0xBAD03605, 0xCDD70693, 0x54DE5729, 0x23D967BF, 0xB3667A2E, 0xC4614AB8, 0x5D681B02, 0x2A6F2B94,
       0xB40BBE37, 0xC30C8EA1, 0x5A05DF1B, 0x2D02EF8D};

   unsigned int crc = ~0u;
   int i;
   for (i = 0; i < len; ++i)
      crc = (crc >> 8) ^ crc_table[buffer[i] ^ (crc & 0xff)];
   return ~crc;
#endif
}

#define stbiw__wpng4(o, a, b, c, d) ((o)[0] = STBIW_UCHAR(a), (o)[1] = STBIW_UCHAR(b), (o)[2] = STBIW_UCHAR(c), (o)[3] = STBIW_UCHAR(d), (o) += 4)
#define stbiw__wp32(data, v) stbiw__wpng4(data, (v) >> 24, (v) >> 16, (v) >> 8, (v));
#define stbiw__wptag(data, s) stbiw__wpng4(data, s[0], s[1], s[2], s[3])

static void stbiw__wpcrc(unsigned char **data, int len) {
   unsigned int crc = stbiw__crc32(*data - len - 4, len + 4);
   stbiw__wp32(*data, crc);
}

static unsigned char stbiw__paeth(int a, int b, int c) {
   int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
   if (pa <= pb && pa <= pc)
      return STBIW_UCHAR(a);
   if (pb <= pc)
      return STBIW_UCHAR(b);
   return STBIW_UCHAR(c);
}

// @OPTIMIZE: provide an option that always forces left-predict or paeth predict
static void stbiw__encode_png_line(unsigned char *pixels, int stride_bytes, int width, int height, int y, int n, int filter_type, signed char *line_buffer) {
   static int mapping[] = {0, 1, 2, 3, 4};
   static int firstmap[] = {0, 1, 0, 5, 6};
   int *mymap = (y != 0) ? mapping : firstmap;
   int i;
   int type = mymap[filter_type];
   unsigned char *z = pixels + stride_bytes * (stbi__flip_vertically_on_write ? height - 1 - y : y);
   int signed_stride = stbi__flip_vertically_on_write ? -stride_bytes : stride_bytes;

   if (type == 0) {
      memcpy(line_buffer, z, width * n);
      return;
   }

   // first loop isn't optimized since it's just one pixel
   for (i = 0; i < n; ++i) {
      switch (type) {
      case 1:
         line_buffer[i] = z[i];
         break;
      case 2:
         line_buffer[i] = z[i] - z[i - signed_stride];
         break;
      case 3:
         line_buffer[i] = z[i] - (z[i - signed_stride] >> 1);
         break;
      case 4:
         line_buffer[i] = (signed char)(z[i] - stbiw__paeth(0, z[i - signed_stride], 0));
         break;
      case 5:
         line_buffer[i] = z[i];
         break;
      case 6:
         line_buffer[i] = z[i];
         break;
      }
   }
   switch (type) {
   case 1:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - z[i - n];
      break;
   case 2:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - z[i - signed_stride];
      break;
   case 3:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - ((z[i - n] + z[i - signed_stride]) >> 1);
      break;
   case 4:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - stbiw__paeth(z[i - n], z[i - signed_stride], z[i - signed_stride - n]);
      break;
   case 5:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - (z[i - n] >> 1);
      break;
   case 6:
      for (i = n; i < width * n; ++i)
         line_buffer[i] = z[i] - stbiw__paeth(z[i - n], 0, 0);
      break;
   }
}

STBIWDEF unsigned char *stbi_write_png_to_mem(const unsigned char *pixels, int stride_bytes, int x, int y, int n, int *out_len) {
   int force_filter = stbi_write_force_png_filter;
   int ctype[5] = {-1, 0, 4, 2, 6};
   unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
   unsigned char *out, *o, *filt, *zlib;
   signed char *line_buffer;
   int j, zlen;

   if (stride_bytes == 0)
      stride_bytes = x * n;

   if (force_filter >= 5) {
      force_filter = -1;
   }

   filt = (unsigned char *)STBIW_MALLOC((x * n + 1) * y);
   if (!filt)
      return 0;
   line_buffer = (signed char *)STBIW_MALLOC(x * n);
   if (!line_buffer) {
      STBIW_FREE(filt);
      return 0;
   }
   for (j = 0; j < y; ++j) {
      int filter_type;
      if (force_filter > -1) {
         filter_type = force_filter;
         stbiw__encode_png_line((unsigned char *)(pixels), stride_bytes, x, y, j, n, force_filter, line_buffer);
      } else { // Estimate the best filter by running through all of them:
         int best_filter = 0, best_filter_val = 0x7fffffff, est, i;
         for (filter_type = 0; filter_type < 5; filter_type++) {
            stbiw__encode_png_line((unsigned char *)(pixels), stride_bytes, x, y, j, n, filter_type, line_buffer);

            // Estimate the entropy of the line using this filter; the less, the better.
            est = 0;
            for (i = 0; i < x * n; ++i) {
               est += abs((signed char)line_buffer[i]);
            }
            if (est < best_filter_val) {
               best_filter_val = est;
               best_filter = filter_type;
            }
         }
         if (filter_type != best_filter) { // If the last iteration already got us the best filter, don't redo it
            stbiw__encode_png_line((unsigned char *)(pixels), stride_bytes, x, y, j, n, best_filter, line_buffer);
            filter_type = best_filter;
         }
      }
      // when we get here, filter_type contains the filter type, and line_buffer contains the data
      filt[j * (x * n + 1)] = (unsigned char)filter_type;
      STBIW_MEMMOVE(filt + j * (x * n + 1) + 1, line_buffer, x * n);
   }
   STBIW_FREE(line_buffer);
   zlib = stbi_zlib_compress(filt, y * (x * n + 1), &zlen, stbi_write_png_compression_level);
   STBIW_FREE(filt);
   if (!zlib)
      return 0;

   // each tag requires 12 bytes of overhead
   out = (unsigned char *)STBIW_MALLOC(8 + 12 + 13 + 12 + zlen + 12);
   if (!out)
      return 0;
   *out_len = 8 + 12 + 13 + 12 + zlen + 12;

   o = out;
   STBIW_MEMMOVE(o, sig, 8);
   o += 8;
   stbiw__wp32(o, 13); // header length
   stbiw__wptag(o, "IHDR");
   stbiw__wp32(o, x);
   stbiw__wp32(o, y);
   *o++ = 8;
   *o++ = STBIW_UCHAR(ctype[n]);
   *o++ = 0;
   *o++ = 0;
   *o++ = 0;
   stbiw__wpcrc(&o, 13);

   stbiw__wp32(o, zlen);
   stbiw__wptag(o, "IDAT");
   STBIW_MEMMOVE(o, zlib, zlen);
   o += zlen;
   STBIW_FREE(zlib);
   stbiw__wpcrc(&o, zlen);

   stbiw__wp32(o, 0);
   stbiw__wptag(o, "IEND");
   stbiw__wpcrc(&o, 0);

   STBIW_ASSERT(o == out + *out_len);

   return out;
}

#ifndef STBI_WRITE_NO_STDIO
STBIWDEF int stbi_write_png(char const *filename, int x, int y, int comp, const void *data, int stride_bytes) {
   FILE *f;
   int len;
   unsigned char *png = stbi_write_png_to_mem((const unsigned char *)data, stride_bytes, x, y, comp, &len);
   if (png == NULL)
      return 0;

   f = stbiw__fopen(filename, "wb");
   if (!f) {
      STBIW_FREE(png);
      return 0;
   }
   fwrite(png, 1, len, f);
   fclose(f);
   STBIW_FREE(png);
   return 1;
}
#endif

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/