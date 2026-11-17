#!/bin/bash
# uso: ingest.sh PREFISSO  -> copia i CSV scaricati dopo l'ultimo marker, nominandoli per anno
D=/root/.claude/projects/-home-claude/a124db30-5f26-5c32-b83d-d31447d5353d/tool-results
O=/home/claude/data/subgiornalieri; M=$O/.marker
for f in $(find $D -name 'mcp-Gmail-blob-*.csv' -newer $M | sort); do
  y=$(sed -n 2p "$f" | cut -d';' -f1 | cut -d- -f3); cp "$f" "$O/$1_$y.csv"; echo "$1_$y $(wc -l < $f)"
done
touch $M
