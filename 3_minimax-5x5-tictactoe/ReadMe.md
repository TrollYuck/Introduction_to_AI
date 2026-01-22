# Tomasz Niedziałek 279754

W powyższym repozytorium znajduje się implementacja algorytmu minmax, zaimplementowanego do gry w warcaby 5x5.
Działa w połączeniu z serwerem napisanym przez prowadzącego przedmiotu - Macieja Gębalę.

Bot w języku Go.

Kompilacja:
go build -o 279754 bot.go board.go node.go

Uruchomienie zgodne z założeniami zadania:
./279754 <numer ip> <numer portu> <gracz> <nick> <depth>