grammar statmt;

declr : whereClause whenClause? '{'
            reduceClause '{' innerOp '}'
            epilogOp?
        '}' EOF;

whereClause : WHERE '(' variable ',' variable ')';
whenClause : WHEN '(' natOp ')';
reduceClause : REDUCE '(' variable ')';

innerOp : array asOp expression;
natOp : NatId '(' arrList ')';
epilogOp : (aggOp | selOp | unlOp);


aggOp : AggId '(' arrList ')';
selOp : SelId '(' expression ')';
unlOp :
UnlId '(' SCIENTIFIC_NUMBER ',' variable ',' 
expression ')';

expression
   :  uop expression
   |  expression mulop expression
   |  expression addop expression
   |  expression shiop expression
   |  expression ueqop expression
   |  expression eqop expression
   |  expression '&' expression
   |  expression '^' expression
   |  expression '|' expression
   |  expression '&&' expression
   |  expression '||' expression
   |  '(' expression ')'
   |  atom
   /*|  copArr*/
   | array
   ;

/*copArr
   : array
   | variable '[' arrList ']'
   ;*/

array
   : variable '[' varList ']'
   ;

varList
   : variable (',' variable)?
   ;

arrList 
   : array (',' array)?
   ;

asOp
   : '+='
   | '-='
   | '*='
   | '/='
   | '='
   ;

atom
   : scientific
   | variable
   ;

scientific
   : SCIENTIFIC_NUMBER
   ;

variable
   : VARIABLE
   ;


uop
   : '!'
   | '~'
   | '-'
   ;

mulop
   : '*'
   | '/'
   | '%'
   ;

addop
   : '+'
   | '-'
   ;

shiop
   : '>>'
   | '<<'
   ;

ueqop
   : '>'
   | '<'
   | '>='
   | '<='
   ;

eqop
   : '=='
   | '!='
   ;

NatId
   : '__natural__'
   ;

AggId
   : '__aggregate__'
   ;

SelId
   : '__select__'
   ;

UnlId
   : '__unlinear__'
   ;

WHERE
   : 'where'
   ;

WHEN
   : 'when'
   ;

REDUCE
   : 'reduce'
   ;

VARIABLE
   : VALID_ID_START VALID_ID_CHAR*
   ;

fragment VALID_ID_START
   : ('a' .. 'z') | ('A' .. 'Z') | '_'
   ;

fragment VALID_ID_CHAR
   : VALID_ID_START | ('0' .. '9')
   ;


SCIENTIFIC_NUMBER
   : SIGN? NUMBER (E SIGN? UNSIGNED_INTEGER)?
   ;

fragment NUMBER
   : ('0' .. '9') + ('.' ('0' .. '9') *)?
   ;

fragment UNSIGNED_INTEGER
   : ('0' .. '9')+
   ;

fragment E
   : 'E' | 'e'
   ;

fragment SIGN
   : ('+' | '-')
   ;


WS
   : [ \r\n\t] + -> skip
   ;