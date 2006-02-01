;; This buffer is for notes you don't want to save, and for Lisp evaluation.
;; If you want to create a file, first visit that file with C-x C-f,
;; then enter the text in that file's own buffer.

(c++-mode)
(mark-whole-buffer)
(c-indent-line-or-region)
(save-buffer)
;;(save-buffers-kill-emacs)
