define viewQString
  set $i=0
  while $i < $arg0.d->len
    printf "%c", $arg0.d->unicode[$i++].ucs
  end
  printf "\n"
end
