%%
%% This is file `ddis_thesis.cls'
%% 
%% Many thanks to Beat Fluri (fluri@ifi.uzh.ch) and s.e.a.l.
%% (seal.ifi.uzh.ch) who kindly made this file available to ddis.
%% 
%%  \CharacterTable
%%  {Upper-case    \A\B\C\D\E\F\G\H\I\J\K\L\M\N\O\P\Q\R\S\T\U\V\W\X\Y\Z
%%   Lower-case    \a\b\c\d\e\f\g\h\i\j\k\l\m\n\o\p\q\r\s\t\u\v\w\x\y\z
%%   Digits        \0\1\2\3\4\5\6\7\8\9
%%   Exclamation   \!     Double quote  \"     Hash (number) \#
%%   Dollar        \$     Percent       \%     Ampersand     \&
%%   Acute accent  \'     Left paren    \(     Right paren   \)
%%   Asterisk      \*     Plus          \+     Comma         \,
%%   Minus         \-     Point         \.     Solidus       \/
%%   Colon         \:     Semicolon     \;     Less than     \<
%%   Equals        \=     Greater than  \>     Question mark \?
%%   Commercial at \@     Left bracket  \[     Backslash     \\
%%   Right bracket \]     Circumflex    \^     Underscore    \_
%%   Grave accent  \`     Left brace    \{     Vertical bar  \|
%%   Right brace   \}     Tilde         \~}

%% Change log
%% 
%% 20/10/05 initial version         Christoph Kiefer (ck)
%% 13/12/05 chapter headings changed    ck
%% 06/06/06 switched to German version  ck
%% 18/05/08 fixed lol spacing           ck
%% 29/11/12 pictured didn't work, used class scrbook, some changes to German need to be done	Mihaela Verman

\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\def\filename{ddis-thesis-DE.cls}
\ProvidesClass{ddis-thesis-DE}
              [2006/06/06 v1.0e
 ddis LaTeX thesis class]

\LoadClass{scrbook}[1997/04/16]
\RequirePackage{listings}


%% LOAD BABEL
\RequirePackage[\sprache]{babel}[1997/01/23]

%% TRY TO LOAD CSQUOTES

%%\IfFileExists{csquotes.sty}{%
%%   \usepackage[babel, german=swiss]{csquotes}
%%   \MakeAutoQuote{�}{�}
%%  \MakeInnerQuote{�}
%%}{}

%% HYPERLINKS
\RequirePackage{url}
\newcommand{\urlprefix}{}
\@ifundefined{selectfont}{%
\def\UrlFont{\it}}
{\def\UrlFont{\rmfamily\itshape}
}

\renewcommand\chapter{\if@openright
                        \clearpage\thispagestyle{empty}\cleardoublepage
                    \else
                        \clearpage
                    \fi
                    \thispagestyle{empty}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \addtocontents{lol}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}

% \def\@makechapterhead#1{%
%   \vspace*{50\p@}%
%   {\parindent \z@ \raggedright
%     \ifnum \c@secnumdepth >\m@ne
%       \if@mainmatter
%         {\fontfamily{pfr}\fontsize{22}{0}\selectfont\bfseries \@chapapp\space \thechapter}%
%         \par\nobreak
%         \vskip 20\p@
%       \fi
%     \fi
%     \interlinepenalty\@M
%     {\fontfamily{pfr}\fontsize{32}{0}\selectfont\bfseries #1}\par\nobreak
%     \vskip 40\p@
%   }}

\def\@makechapterhead#1{%
  \reset@font
  \vspace*{10\p@}%
  {\parindent \z@ 
    \begin{flushleft}
%       \reset@font \scshape \bfseries \Huge \thechapter \par
        \reset@font {\fontfamily{pfr}\fontsize{72}{0}\selectfont\bfseries \thechapter} \par
    \end{flushleft}
%    \hrule
     \vskip 20\p@
    \begin{flushleft}
%       \reset@font \LARGE \strut #1\strut \par
        \reset@font {\fontfamily{pfr}\fontsize{32}{0}\selectfont\bfseries #1} \strut \par
    \end{flushleft}
%     \vskip 100\p@
      \vskip 40\p@
  }}

\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \interlinepenalty\@M
    \vbox{\hrule height0.4pt width\textwidth}
    \vskip 10\p@
    {\fontfamily{pfr}\fontsize{32}{0}\selectfont\bfseries #1}\par\nobreak
    \vskip 40\p@
  }}

\renewcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.8ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\fontfamily{pfr}\selectfont\LARGE\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.3ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\fontfamily{pfr}\selectfont\Large\bfseries}}
\renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.3ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\fontfamily{pfr}\selectfont\large\bfseries}}
\renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}

\renewcommand\labelitemi{\tiny\raisebox{1.5pt}{$\m@th\bullet$}}

\def\fnum@figure{{\bfseries\fontfamily{pfr}\selectfont \figurename\nobreakspace\thefigure}}
\def\fnum@table{{\bfseries\fontfamily{pfr}\selectfont \tablename\nobreakspace\thetable}}

% Original name = Contents
\renewcommand{\contentsname}{Table of Contents}
% Original name = Bibliography
\renewcommand\bibname{References}

% new contenttable for lists and tables
%\renewcommand\listoffigures{%
%    \newpage
%    \section*{\listfigurename}
%        \@starttoc{lof}%
%}
%\renewcommand\listoftables{%
%    \section*{\listtablename}
%        \@starttoc{lot}%
%}
\renewcommand\lstlistoflistings{%
    \chapter*{List of \lstlistlistingname}
        \@starttoc{lol}%
}

\newenvironment{abstract}{%
  \chapter*{Abstract}}
  {}

\newenvironment{zusammenfassung}{%
  \chapter*{Zusammenfassung}}{}

\newenvironment{acknowledgements}{%
  \chapter*{Danksagung}}{}



\endinput
