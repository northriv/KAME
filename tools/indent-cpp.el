;; This buffer is for notes you don't want to save, and for Lisp evaluation.
;; If you want to create a file, first visit that file with C-x C-f,
;; then enter the text in that file's own buffer.
;; C/C++ �p�̃J�X�^�}�C�Y
(defconst my-c-style
  '(
    ;; ��{�I�t�Z�b�g�ʂ̐ݒ�
    (c-basic-offset             . 4)

    ;; tab �L�[�ŃC���f���g�����s
    (c-tab-always-indent        . t)

    ;; �R�����g�s�̃I�t�Z�b�g�ʂ̐ݒ�
    (c-comment-only-line-offset . 0)

    ;; �J�b�R�O��̎������s�����̐ݒ�
    (c-hanging-braces-alist
     . (
        (class-open before after)       ; �N���X�錾��'{'�̑O��
        (class-close after)             ; �N���X�錾��'}'�̌�
        (defun-open before after)       ; �֐��錾��'{'�̑O��
        (defun-close after)             ; �֐��錾��'}'�̌�
        (inline-open after)             ; �N���X���̃C�����C��
                                        ; �֐��錾��'{'�̌�
        (inline-close after)            ; �N���X���̃C�����C��
                                        ; �֐��錾��'}'�̌�
        (brace-list-open after)         ; �񋓌^�A�z��錾��'{'�̌�
        (brace-list-close before after) ; �񋓌^�A�z��錾��'}'�̑O��
        (block-open after)              ; �X�e�[�g�����g��'{'�̌�
        (block-close after)             ; �X�e�[�g�����g��'}'�O��
        (substatement-open after)       ; �T�u�X�e�[�g�����g
                                        ; (if ����)��'{'�̌�
        (statement-case-open after)     ; case ����'{'�̌�
        (extern-lang-open before after) ; ������ւ̃����P�[�W�錾��
                                        ; '{'�̑O��
        (extern-lang-close before)      ; ������ւ̃����P�[�W�錾��
                                        ; '}'�̑O
        ))

    ;; �R�����O��̎������s�����̐ݒ�
    (c-hanging-colons-alist
     . (
        (case-label after)              ; case ���x����':'�̌�
        (label after)                   ; ���x����':'�̌�
        (access-label after)            ; �A�N�Z�X���x��(public��)��':'�̌�
        (member-init-intro)             ; �R���X�g���N�^�ł̃����o�[������
                                        ; ���X�g�̐擪��':'�ł͉��s���Ȃ�
        (inher-intro before)            ; �N���X�錾�ł̌p�����X�g�̐擪��
                                        ; ':'�ł͉��s���Ȃ�
        ))

    ;; �}�����ꂽ�]�v�ȋ󔒕����̃L�����Z�������̐ݒ�
    ;; ���L��*���폜����
    (c-cleanup-list
     . (
        brace-else-brace                ; else �̒��O
                                        ; "} * else {"  ->  "} else {"
        brace-elseif-brace              ; else if �̒��O
                                        ; "} * else if (.*) {"
                                        ; ->  } "else if (.*) {"
        empty-defun-braces              ; ��̃N���X�E�֐���`��'}' �̒��O
                                        ;�G"{ * }"  ->  "{}"
        defun-close-semi                ; �N���X�E�֐���`���';' �̒��O
                                        ; "} * ;"  ->  "};"
        list-close-comma                ; �z�񏉊�������'},'�̒��O
                                        ; "} * ,"  ->  "},"
        scope-operator                  ; �X�R�[�v���Z�q'::' �̊�
                                        ; ": * :"  ->  "::"
        ))

    ;; �I�t�Z�b�g�ʂ̐ݒ�
    ;; �K�v�����̂ݔ���(���̐ݒ�ɕt���Ă� info �Q��)
    ;; �I�t�Z�b�g�ʂ͉��L�Ŏw��
    ;; +  c-basic-offset�� 1�{, ++ c-basic-offset�� 2�{
    ;; -  c-basic-offset��-1�{, -- c-basic-offset��-2�{
    (c-offsets-alist
     . (
        (arglist-intro          . 4)   ; �������X�g�̊J�n�s
        (arglist-close          . c-lineup-arglist) ; �������X�g�̏I���s
        (substatement-open      . 0)    ; �T�u�X�e�[�g�����g�̊J�n�s
        (statement-cont         . 4)   ; �X�e�[�g�����g�̌p���s
        (case-label             . 0)    ; case ���̃��x���s
        (label                  . 0)    ; ���x���s
        (block-open             . 0)    ; �u���b�N�̊J�n�s
        (member-init-intro      . 4)   ; �����o�I�u�W�F�N�g�̏��������X�g
        ))

    ;; �C���f���g���ɍ\����͏���\������
    (c-echo-syntactic-information-p . t)
    )
  "My C Programming Style")

;; hook �p�̊֐��̒�`
(defun my-c-mode-common-hook ()
  ;; my-c-stye ��o�^���ėL���ɂ���
  (c-add-style "PERSONAL" my-c-style t)

  ;; ���̃X�^�C�����f�t�H���g�ŗp�ӂ���Ă���̂őI�����Ă��悢
;  (c-set-style "gnu")
;  (c-set-style "cc-mode")
;  (c-set-style "stroustrup")
;  (c-set-style "ellemtel")
  ;; �����̃X�^�C����ύX����ꍇ�͎��̂悤�ɂ���
;  (c-set-offset 'member-init-intro '++)

  ;; �^�u���̐ݒ�
  (setq tab-width 4)

  ;; �^�u�̑���ɃX�y�[�X���g��
;  (setq indent-tabs-mode nil)

  ;; �������s(auto-newline)��L���ɂ���
  (c-toggle-auto-state t)

  ;; �A������󔒂̈ꊇ�폜(hungry-delete)��L���ɂ���
  (c-toggle-hungry-state t)

  ;; �Z�~�R�����Ŏ������s���Ȃ�
  (setq c-hanging-semi&comma-criteria nil)
  )
;; ���[�h�ɓ���Ƃ��ɌĂяo�� hook �̐ݒ�
(add-hook 'c-mode-common-hook 'my-c-mode-common-hook)

;; �w�b�_�t�@�C����c++���[�h�ŊJ�� 
(setq auto-mode-alist (append
                       '(("\\.h$"    . c++-mode))
                       auto-mode-alist))
(c++-mode)
(mark-whole-buffer)
(c-indent-line-or-region)

(save-buffer)
;;(save-buffers-kill-emacs)
