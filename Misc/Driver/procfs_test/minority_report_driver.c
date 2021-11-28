/**
 * @file minority_report_driver.c
 * @author David González León, Jade Gröli
 * @brief A test of a driver using the proc_fs module of the linux kernel
 * @version 0.1
 * @date 19-11-2021
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

#include <asm/io.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>

static struct proc_dir_entry *lll_proc = NULL;

ssize_t lll_read(struct file *file, char __user *user, size_t size, loff_t *off) { return 0; }

ssize_t lll_write(struct file *file, const char __user *user, size_t size, loff_t *off) { return 0; }

static const struct proc_ops lll_proc_fops = {
    .proc_read = lll_read,
    .proc_write = lll_write,
};

static int __init gpio_driver_init(void) {
   printk("Welcome to my driver!\n");

   // create an entry in the proc-fs
   lll_proc = proc_create("minority-report", 0666, NULL, &lll_proc_fops);
   if (lll_proc == NULL) {
      return -1;
   }

   return 0;
}

static void __exit gpio_driver_exit(void) {
   printk("Leaving my driver!\n");
   proc_remove(lll_proc);
   return;
}

module_init(gpio_driver_init);
module_exit(gpio_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("David González León, Jade Gröli");
MODULE_DESCRIPTION("Driver for Minority report project");
MODULE_VERSION("0.1");