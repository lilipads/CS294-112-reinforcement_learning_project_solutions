ТЄ
Пв
:
Add
x"T
y"T
z"T"
Ttype:
2	
о
	ApplyAdam
var"TА	
m"TА	
v"TА
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"TА" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"TА

value"T

output_ref"TА"	
Ttype"
validate_shapebool("
use_lockingbool(Ш
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
Р
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	Р
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
М
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtypeА"
shapeshape"
dtypetype"
	containerstring "
shared_namestring И"serve*1.11.02v1.11.0-rc2-4-gc19e29306cќх
t
input_placeholderPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
t
label_placeholderPlaceholder*
dtype0*'
_output_shapes
:€€€€€€€€€*
shape:€€€€€€€€€
µ
8fully_connected/weights/Initializer/random_uniform/shapeConst*
valueB"   @   **
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
:
І
6fully_connected/weights/Initializer/random_uniform/minConst*
valueB
 *HYЛЊ**
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
: 
І
6fully_connected/weights/Initializer/random_uniform/maxConst*
valueB
 *HYЛ>**
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
: 
Ж
@fully_connected/weights/Initializer/random_uniform/RandomUniformRandomUniform8fully_connected/weights/Initializer/random_uniform/shape*

seed *
T0**
_class 
loc:@fully_connected/weights*
seed2 *
dtype0*
_output_shapes

:@
ъ
6fully_connected/weights/Initializer/random_uniform/subSub6fully_connected/weights/Initializer/random_uniform/max6fully_connected/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes
: 
М
6fully_connected/weights/Initializer/random_uniform/mulMul@fully_connected/weights/Initializer/random_uniform/RandomUniform6fully_connected/weights/Initializer/random_uniform/sub*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
ю
2fully_connected/weights/Initializer/random_uniformAdd6fully_connected/weights/Initializer/random_uniform/mul6fully_connected/weights/Initializer/random_uniform/min*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
Ј
fully_connected/weights
VariableV2*
dtype0*
_output_shapes

:@*
shared_name **
_class 
loc:@fully_connected/weights*
	container *
shape
:@
у
fully_connected/weights/AssignAssignfully_connected/weights2fully_connected/weights/Initializer/random_uniform*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@
Ц
fully_connected/weights/readIdentityfully_connected/weights*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
®
7fully_connected/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *oГ:**
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
: 
≠
8fully_connected/kernel/Regularizer/l2_regularizer/L2LossL2Lossfully_connected/weights/read*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes
: 
ш
1fully_connected/kernel/Regularizer/l2_regularizerMul7fully_connected/kernel/Regularizer/l2_regularizer/scale8fully_connected/kernel/Regularizer/l2_regularizer/L2Loss*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes
: 
†
(fully_connected/biases/Initializer/zerosConst*
valueB@*    *)
_class
loc:@fully_connected/biases*
dtype0*
_output_shapes
:@
≠
fully_connected/biases
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *)
_class
loc:@fully_connected/biases*
	container *
shape:@
в
fully_connected/biases/AssignAssignfully_connected/biases(fully_connected/biases/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
:@
П
fully_connected/biases/readIdentityfully_connected/biases*
_output_shapes
:@*
T0*)
_class
loc:@fully_connected/biases
©
fully_connected/MatMulMatMulinput_placeholderfully_connected/weights/read*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€@*
transpose_b( 
†
fully_connected/BiasAddBiasAddfully_connected/MatMulfully_connected/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
g
fully_connected/TanhTanhfully_connected/BiasAdd*
T0*'
_output_shapes
:€€€€€€€€€@
є
:fully_connected_1/weights/Initializer/random_uniform/shapeConst*
valueB"@   @   *,
_class"
 loc:@fully_connected_1/weights*
dtype0*
_output_shapes
:
Ђ
8fully_connected_1/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *„≥]Њ*,
_class"
 loc:@fully_connected_1/weights
Ђ
8fully_connected_1/weights/Initializer/random_uniform/maxConst*
valueB
 *„≥]>*,
_class"
 loc:@fully_connected_1/weights*
dtype0*
_output_shapes
: 
М
Bfully_connected_1/weights/Initializer/random_uniform/RandomUniformRandomUniform:fully_connected_1/weights/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed *
T0*,
_class"
 loc:@fully_connected_1/weights*
seed2 
В
8fully_connected_1/weights/Initializer/random_uniform/subSub8fully_connected_1/weights/Initializer/random_uniform/max8fully_connected_1/weights/Initializer/random_uniform/min*
_output_shapes
: *
T0*,
_class"
 loc:@fully_connected_1/weights
Ф
8fully_connected_1/weights/Initializer/random_uniform/mulMulBfully_connected_1/weights/Initializer/random_uniform/RandomUniform8fully_connected_1/weights/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes

:@@
Ж
4fully_connected_1/weights/Initializer/random_uniformAdd8fully_connected_1/weights/Initializer/random_uniform/mul8fully_connected_1/weights/Initializer/random_uniform/min*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes

:@@
ї
fully_connected_1/weights
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *,
_class"
 loc:@fully_connected_1/weights*
	container *
shape
:@@
ы
 fully_connected_1/weights/AssignAssignfully_connected_1/weights4fully_connected_1/weights/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*,
_class"
 loc:@fully_connected_1/weights
Ь
fully_connected_1/weights/readIdentityfully_connected_1/weights*
_output_shapes

:@@*
T0*,
_class"
 loc:@fully_connected_1/weights
ђ
9fully_connected_1/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *oГ:*,
_class"
 loc:@fully_connected_1/weights*
dtype0*
_output_shapes
: 
≥
:fully_connected_1/kernel/Regularizer/l2_regularizer/L2LossL2Lossfully_connected_1/weights/read*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes
: 
А
3fully_connected_1/kernel/Regularizer/l2_regularizerMul9fully_connected_1/kernel/Regularizer/l2_regularizer/scale:fully_connected_1/kernel/Regularizer/l2_regularizer/L2Loss*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes
: 
§
*fully_connected_1/biases/Initializer/zerosConst*
valueB@*    *+
_class!
loc:@fully_connected_1/biases*
dtype0*
_output_shapes
:@
±
fully_connected_1/biases
VariableV2*
shared_name *+
_class!
loc:@fully_connected_1/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
к
fully_connected_1/biases/AssignAssignfully_connected_1/biases*fully_connected_1/biases/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@fully_connected_1/biases*
validate_shape(*
_output_shapes
:@
Х
fully_connected_1/biases/readIdentityfully_connected_1/biases*
T0*+
_class!
loc:@fully_connected_1/biases*
_output_shapes
:@
∞
fully_connected_1/MatMulMatMulfully_connected/Tanhfully_connected_1/weights/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
¶
fully_connected_1/BiasAddBiasAddfully_connected_1/MatMulfully_connected_1/biases/read*
T0*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€@
k
fully_connected_1/TanhTanhfully_connected_1/BiasAdd*'
_output_shapes
:€€€€€€€€€@*
T0
є
:fully_connected_2/weights/Initializer/random_uniform/shapeConst*
valueB"@      *,
_class"
 loc:@fully_connected_2/weights*
dtype0*
_output_shapes
:
Ђ
8fully_connected_2/weights/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *феХЊ*,
_class"
 loc:@fully_connected_2/weights
Ђ
8fully_connected_2/weights/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *феХ>*,
_class"
 loc:@fully_connected_2/weights
М
Bfully_connected_2/weights/Initializer/random_uniform/RandomUniformRandomUniform:fully_connected_2/weights/Initializer/random_uniform/shape*
seed2 *
dtype0*
_output_shapes

:@*

seed *
T0*,
_class"
 loc:@fully_connected_2/weights
В
8fully_connected_2/weights/Initializer/random_uniform/subSub8fully_connected_2/weights/Initializer/random_uniform/max8fully_connected_2/weights/Initializer/random_uniform/min*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes
: 
Ф
8fully_connected_2/weights/Initializer/random_uniform/mulMulBfully_connected_2/weights/Initializer/random_uniform/RandomUniform8fully_connected_2/weights/Initializer/random_uniform/sub*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes

:@
Ж
4fully_connected_2/weights/Initializer/random_uniformAdd8fully_connected_2/weights/Initializer/random_uniform/mul8fully_connected_2/weights/Initializer/random_uniform/min*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes

:@
ї
fully_connected_2/weights
VariableV2*
shared_name *,
_class"
 loc:@fully_connected_2/weights*
	container *
shape
:@*
dtype0*
_output_shapes

:@
ы
 fully_connected_2/weights/AssignAssignfully_connected_2/weights4fully_connected_2/weights/Initializer/random_uniform*
use_locking(*
T0*,
_class"
 loc:@fully_connected_2/weights*
validate_shape(*
_output_shapes

:@
Ь
fully_connected_2/weights/readIdentityfully_connected_2/weights*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes

:@
ђ
9fully_connected_2/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *oГ:*,
_class"
 loc:@fully_connected_2/weights*
dtype0*
_output_shapes
: 
≥
:fully_connected_2/kernel/Regularizer/l2_regularizer/L2LossL2Lossfully_connected_2/weights/read*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes
: 
А
3fully_connected_2/kernel/Regularizer/l2_regularizerMul9fully_connected_2/kernel/Regularizer/l2_regularizer/scale:fully_connected_2/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*,
_class"
 loc:@fully_connected_2/weights
§
*fully_connected_2/biases/Initializer/zerosConst*
valueB*    *+
_class!
loc:@fully_connected_2/biases*
dtype0*
_output_shapes
:
±
fully_connected_2/biases
VariableV2*
shared_name *+
_class!
loc:@fully_connected_2/biases*
	container *
shape:*
dtype0*
_output_shapes
:
к
fully_connected_2/biases/AssignAssignfully_connected_2/biases*fully_connected_2/biases/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@fully_connected_2/biases*
validate_shape(*
_output_shapes
:
Х
fully_connected_2/biases/readIdentityfully_connected_2/biases*
T0*+
_class!
loc:@fully_connected_2/biases*
_output_shapes
:
≤
fully_connected_2/MatMulMatMulfully_connected_1/Tanhfully_connected_2/weights/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
¶
fully_connected_2/BiasAddBiasAddfully_connected_2/MatMulfully_connected_2/biases/read*
data_formatNHWC*'
_output_shapes
:€€€€€€€€€*
T0
Щ
$mean_squared_error/SquaredDifferenceSquaredDifferencefully_connected_2/BiasAddlabel_placeholder*
T0*'
_output_shapes
:€€€€€€€€€
t
/mean_squared_error/assert_broadcastable/weightsConst*
valueB
 *  А?*
dtype0*
_output_shapes
: 
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
Ш
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
u
3mean_squared_error/assert_broadcastable/values/rankConst*
value	B :*
dtype0*
_output_shapes
: 
K
Cmean_squared_error/assert_broadcastable/static_scalar_check_successNoOp
І
mean_squared_error/ToFloat/xConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
У
mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat/x*'
_output_shapes
:€€€€€€€€€*
T0
ѓ
mean_squared_error/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
Н
mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/Const*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
±
&mean_squared_error/num_present/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *    
Ф
$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat/x&mean_squared_error/num_present/Equal/y*
T0*
_output_shapes
: 
і
)mean_squared_error/num_present/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ј
.mean_squared_error/num_present/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
є
.mean_squared_error/num_present/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
√
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
Ћ
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
№
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
Џ
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
value	B : 
ь
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ў
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
value	B :*
dtype0*
_output_shapes
: 
ѓ
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_successNoOpD^mean_squared_error/assert_broadcastable/static_scalar_check_success
ќ
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifferenceD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
T0*
out_type0*
_output_shapes
:
ѓ
@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_successb^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB
 *  А?
К
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*'
_output_shapes
:€€€€€€€€€
ћ
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
T0*'
_output_shapes
:€€€€€€€€€
ї
$mean_squared_error/num_present/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB"       *
dtype0*
_output_shapes
:
ї
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
£
mean_squared_error/Const_1ConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
dtype0*
_output_shapes
: *
valueB 
С
mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const_1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
І
mean_squared_error/Greater/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Д
mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
_output_shapes
: *
T0
•
mean_squared_error/Equal/yConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
_output_shapes
: *
T0
Ђ
"mean_squared_error/ones_like/ShapeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB *
dtype0*
_output_shapes
: 
≠
"mean_squared_error/ones_like/ConstConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *  А?*
dtype0*
_output_shapes
: 
Я
mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
Ь
mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
T0*
_output_shapes
: 
®
mean_squared_error/zeros_likeConstD^mean_squared_error/assert_broadcastable/static_scalar_check_success*
valueB
 *    *
dtype0*
_output_shapes
: 
Ц
mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  А?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
w
2gradients/mean_squared_error/value_grad/zeros_likeConst*
dtype0*
_output_shapes
: *
valueB
 *    
є
.gradients/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greatergradients/Fill2gradients/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
ї
0gradients/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater2gradients/mean_squared_error/value_grad/zeros_likegradients/Fill*
T0*
_output_shapes
: 
§
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp/^gradients/mean_squared_error/value_grad/Select1^gradients/mean_squared_error/value_grad/Select_1
Ы
@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity.gradients/mean_squared_error/value_grad/Select9^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/mean_squared_error/value_grad/Select*
_output_shapes
: 
°
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity0gradients/mean_squared_error/value_grad/Select_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/mean_squared_error/value_grad/Select_1*
_output_shapes
: 
n
+gradients/mean_squared_error/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
-gradients/mean_squared_error/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
н
;gradients/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/div_grad/Shape-gradients/mean_squared_error/div_grad/Shape_1*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€*
T0
ґ
-gradients/mean_squared_error/div_grad/RealDivRealDiv@gradients/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
T0*
_output_shapes
: 
Џ
)gradients/mean_squared_error/div_grad/SumSum-gradients/mean_squared_error/div_grad/RealDiv;gradients/mean_squared_error/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
њ
-gradients/mean_squared_error/div_grad/ReshapeReshape)gradients/mean_squared_error/div_grad/Sum+gradients/mean_squared_error/div_grad/Shape*
_output_shapes
: *
T0*
Tshape0
k
)gradients/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
°
/gradients/mean_squared_error/div_grad/RealDiv_1RealDiv)gradients/mean_squared_error/div_grad/Negmean_squared_error/Select*
T0*
_output_shapes
: 
І
/gradients/mean_squared_error/div_grad/RealDiv_2RealDiv/gradients/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
ƒ
)gradients/mean_squared_error/div_grad/mulMul@gradients/mean_squared_error/value_grad/tuple/control_dependency/gradients/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
Џ
+gradients/mean_squared_error/div_grad/Sum_1Sum)gradients/mean_squared_error/div_grad/mul=gradients/mean_squared_error/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
≈
/gradients/mean_squared_error/div_grad/Reshape_1Reshape+gradients/mean_squared_error/div_grad/Sum_1-gradients/mean_squared_error/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
†
6gradients/mean_squared_error/div_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/div_grad/Reshape0^gradients/mean_squared_error/div_grad/Reshape_1
Х
>gradients/mean_squared_error/div_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/div_grad/Reshape7^gradients/mean_squared_error/div_grad/tuple/group_deps*
_output_shapes
: *
T0*@
_class6
42loc:@gradients/mean_squared_error/div_grad/Reshape
Ы
@gradients/mean_squared_error/div_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/div_grad/Reshape_17^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/div_grad/Reshape_1*
_output_shapes
: 
x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
а
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape>gradients/mean_squared_error/div_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
«
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 
Д
3gradients/mean_squared_error/Sum_grad/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
“
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
Б
+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*
T0*
out_type0*
_output_shapes
:
“
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape+gradients/mean_squared_error/Sum_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:€€€€€€€€€
П
+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*
T0*
out_type0*
_output_shapes
:
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
н
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ђ
)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat/x*
T0*'
_output_shapes
:€€€€€€€€€
Ў
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
–
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*'
_output_shapes
:€€€€€€€€€*
T0*
Tshape0
ґ
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*'
_output_shapes
:€€€€€€€€€*
T0
ё
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
≈
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
†
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1
¶
>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ы
@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
_output_shapes
: 
Т
9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapefully_connected_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
М
;gradients/mean_squared_error/SquaredDifference_grad/Shape_1Shapelabel_placeholder*
T0*
out_type0*
_output_shapes
:
Ч
Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*2
_output_shapes 
:€€€€€€€€€:€€€€€€€€€
ј
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
м
7gradients/mean_squared_error/SquaredDifference_grad/mulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
я
7gradients/mean_squared_error/SquaredDifference_grad/subSubfully_connected_2/BiasAddlabel_placeholder?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€
д
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
T0*'
_output_shapes
:€€€€€€€€€
Д
7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ъ
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
И
9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
А
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:€€€€€€€€€
ѓ
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*'
_output_shapes
:€€€€€€€€€
ƒ
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
ё
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
Ў
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*'
_output_shapes
:€€€€€€€€€
Ќ
4gradients/fully_connected_2/BiasAdd_grad/BiasAddGradBiasAddGradLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
:*
T0
«
9gradients/fully_connected_2/BiasAdd_grad/tuple/group_depsNoOp5^gradients/fully_connected_2/BiasAdd_grad/BiasAddGradM^gradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency
ў
Agradients/fully_connected_2/BiasAdd_grad/tuple/control_dependencyIdentityLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency:^gradients/fully_connected_2/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*'
_output_shapes
:€€€€€€€€€
ѓ
Cgradients/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/fully_connected_2/BiasAdd_grad/BiasAddGrad:^gradients/fully_connected_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients/fully_connected_2/BiasAdd_grad/BiasAddGrad
у
.gradients/fully_connected_2/MatMul_grad/MatMulMatMulAgradients/fully_connected_2/BiasAdd_grad/tuple/control_dependencyfully_connected_2/weights/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€@
д
0gradients/fully_connected_2/MatMul_grad/MatMul_1MatMulfully_connected_1/TanhAgradients/fully_connected_2/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
§
8gradients/fully_connected_2/MatMul_grad/tuple/group_depsNoOp/^gradients/fully_connected_2/MatMul_grad/MatMul1^gradients/fully_connected_2/MatMul_grad/MatMul_1
ђ
@gradients/fully_connected_2/MatMul_grad/tuple/control_dependencyIdentity.gradients/fully_connected_2/MatMul_grad/MatMul9^gradients/fully_connected_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/fully_connected_2/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
©
Bgradients/fully_connected_2/MatMul_grad/tuple/control_dependency_1Identity0gradients/fully_connected_2/MatMul_grad/MatMul_19^gradients/fully_connected_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/fully_connected_2/MatMul_grad/MatMul_1*
_output_shapes

:@
∆
.gradients/fully_connected_1/Tanh_grad/TanhGradTanhGradfully_connected_1/Tanh@gradients/fully_connected_2/MatMul_grad/tuple/control_dependency*'
_output_shapes
:€€€€€€€€€@*
T0
ѓ
4gradients/fully_connected_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/fully_connected_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
©
9gradients/fully_connected_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients/fully_connected_1/BiasAdd_grad/BiasAddGrad/^gradients/fully_connected_1/Tanh_grad/TanhGrad
Ѓ
Agradients/fully_connected_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/fully_connected_1/Tanh_grad/TanhGrad:^gradients/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/fully_connected_1/Tanh_grad/TanhGrad*'
_output_shapes
:€€€€€€€€€@
ѓ
Cgradients/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients/fully_connected_1/BiasAdd_grad/BiasAddGrad:^gradients/fully_connected_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/fully_connected_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
у
.gradients/fully_connected_1/MatMul_grad/MatMulMatMulAgradients/fully_connected_1/BiasAdd_grad/tuple/control_dependencyfully_connected_1/weights/read*
transpose_a( *'
_output_shapes
:€€€€€€€€€@*
transpose_b(*
T0
в
0gradients/fully_connected_1/MatMul_grad/MatMul_1MatMulfully_connected/TanhAgradients/fully_connected_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

:@@*
transpose_b( *
T0
§
8gradients/fully_connected_1/MatMul_grad/tuple/group_depsNoOp/^gradients/fully_connected_1/MatMul_grad/MatMul1^gradients/fully_connected_1/MatMul_grad/MatMul_1
ђ
@gradients/fully_connected_1/MatMul_grad/tuple/control_dependencyIdentity.gradients/fully_connected_1/MatMul_grad/MatMul9^gradients/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/fully_connected_1/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€@
©
Bgradients/fully_connected_1/MatMul_grad/tuple/control_dependency_1Identity0gradients/fully_connected_1/MatMul_grad/MatMul_19^gradients/fully_connected_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/fully_connected_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
¬
,gradients/fully_connected/Tanh_grad/TanhGradTanhGradfully_connected/Tanh@gradients/fully_connected_1/MatMul_grad/tuple/control_dependency*
T0*'
_output_shapes
:€€€€€€€€€@
Ђ
2gradients/fully_connected/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/fully_connected/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes
:@
£
7gradients/fully_connected/BiasAdd_grad/tuple/group_depsNoOp3^gradients/fully_connected/BiasAdd_grad/BiasAddGrad-^gradients/fully_connected/Tanh_grad/TanhGrad
¶
?gradients/fully_connected/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/fully_connected/Tanh_grad/TanhGrad8^gradients/fully_connected/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:€€€€€€€€€@*
T0*?
_class5
31loc:@gradients/fully_connected/Tanh_grad/TanhGrad
І
Agradients/fully_connected/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/fully_connected/BiasAdd_grad/BiasAddGrad8^gradients/fully_connected/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/fully_connected/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
н
,gradients/fully_connected/MatMul_grad/MatMulMatMul?gradients/fully_connected/BiasAdd_grad/tuple/control_dependencyfully_connected/weights/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:€€€€€€€€€
џ
.gradients/fully_connected/MatMul_grad/MatMul_1MatMulinput_placeholder?gradients/fully_connected/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes

:@*
transpose_b( 
Ю
6gradients/fully_connected/MatMul_grad/tuple/group_depsNoOp-^gradients/fully_connected/MatMul_grad/MatMul/^gradients/fully_connected/MatMul_grad/MatMul_1
§
>gradients/fully_connected/MatMul_grad/tuple/control_dependencyIdentity,gradients/fully_connected/MatMul_grad/MatMul7^gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/fully_connected/MatMul_grad/MatMul*'
_output_shapes
:€€€€€€€€€
°
@gradients/fully_connected/MatMul_grad/tuple/control_dependency_1Identity.gradients/fully_connected/MatMul_grad/MatMul_17^gradients/fully_connected/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/fully_connected/MatMul_grad/MatMul_1*
_output_shapes

:@
Й
beta1_power/initial_valueConst*
valueB
 *fff?*)
_class
loc:@fully_connected/biases*
dtype0*
_output_shapes
: 
Ъ
beta1_power
VariableV2*)
_class
loc:@fully_connected/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
є
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
: 
u
beta1_power/readIdentitybeta1_power*
T0*)
_class
loc:@fully_connected/biases*
_output_shapes
: 
Й
beta2_power/initial_valueConst*
valueB
 *wЊ?*)
_class
loc:@fully_connected/biases*
dtype0*
_output_shapes
: 
Ъ
beta2_power
VariableV2*)
_class
loc:@fully_connected/biases*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
є
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
: 
u
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*)
_class
loc:@fully_connected/biases
ї
>fully_connected/weights/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   @   **
_class 
loc:@fully_connected/weights
•
4fully_connected/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
: 
У
.fully_connected/weights/Adam/Initializer/zerosFill>fully_connected/weights/Adam/Initializer/zeros/shape_as_tensor4fully_connected/weights/Adam/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
Љ
fully_connected/weights/Adam
VariableV2*
shared_name **
_class 
loc:@fully_connected/weights*
	container *
shape
:@*
dtype0*
_output_shapes

:@
щ
#fully_connected/weights/Adam/AssignAssignfully_connected/weights/Adam.fully_connected/weights/Adam/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@
†
!fully_connected/weights/Adam/readIdentityfully_connected/weights/Adam*
_output_shapes

:@*
T0**
_class 
loc:@fully_connected/weights
љ
@fully_connected/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"   @   **
_class 
loc:@fully_connected/weights
І
6fully_connected/weights/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    **
_class 
loc:@fully_connected/weights*
dtype0*
_output_shapes
: 
Щ
0fully_connected/weights/Adam_1/Initializer/zerosFill@fully_connected/weights/Adam_1/Initializer/zeros/shape_as_tensor6fully_connected/weights/Adam_1/Initializer/zeros/Const*
T0*

index_type0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
Њ
fully_connected/weights/Adam_1
VariableV2*
shared_name **
_class 
loc:@fully_connected/weights*
	container *
shape
:@*
dtype0*
_output_shapes

:@
€
%fully_connected/weights/Adam_1/AssignAssignfully_connected/weights/Adam_10fully_connected/weights/Adam_1/Initializer/zeros*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@
§
#fully_connected/weights/Adam_1/readIdentityfully_connected/weights/Adam_1*
T0**
_class 
loc:@fully_connected/weights*
_output_shapes

:@
•
-fully_connected/biases/Adam/Initializer/zerosConst*
valueB@*    *)
_class
loc:@fully_connected/biases*
dtype0*
_output_shapes
:@
≤
fully_connected/biases/Adam
VariableV2*)
_class
loc:@fully_connected/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
с
"fully_connected/biases/Adam/AssignAssignfully_connected/biases/Adam-fully_connected/biases/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
Щ
 fully_connected/biases/Adam/readIdentityfully_connected/biases/Adam*
T0*)
_class
loc:@fully_connected/biases*
_output_shapes
:@
І
/fully_connected/biases/Adam_1/Initializer/zerosConst*
valueB@*    *)
_class
loc:@fully_connected/biases*
dtype0*
_output_shapes
:@
і
fully_connected/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *)
_class
loc:@fully_connected/biases*
	container *
shape:@
ч
$fully_connected/biases/Adam_1/AssignAssignfully_connected/biases/Adam_1/fully_connected/biases/Adam_1/Initializer/zeros*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
Э
"fully_connected/biases/Adam_1/readIdentityfully_connected/biases/Adam_1*
T0*)
_class
loc:@fully_connected/biases*
_output_shapes
:@
њ
@fully_connected_1/weights/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@   @   *,
_class"
 loc:@fully_connected_1/weights
©
6fully_connected_1/weights/Adam/Initializer/zeros/ConstConst*
valueB
 *    *,
_class"
 loc:@fully_connected_1/weights*
dtype0*
_output_shapes
: 
Ы
0fully_connected_1/weights/Adam/Initializer/zerosFill@fully_connected_1/weights/Adam/Initializer/zeros/shape_as_tensor6fully_connected_1/weights/Adam/Initializer/zeros/Const*
T0*

index_type0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes

:@@
ј
fully_connected_1/weights/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *,
_class"
 loc:@fully_connected_1/weights*
	container *
shape
:@@
Б
%fully_connected_1/weights/Adam/AssignAssignfully_connected_1/weights/Adam0fully_connected_1/weights/Adam/Initializer/zeros*
T0*,
_class"
 loc:@fully_connected_1/weights*
validate_shape(*
_output_shapes

:@@*
use_locking(
¶
#fully_connected_1/weights/Adam/readIdentityfully_connected_1/weights/Adam*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes

:@@
Ѕ
Bfully_connected_1/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@   @   *,
_class"
 loc:@fully_connected_1/weights*
dtype0*
_output_shapes
:
Ђ
8fully_connected_1/weights/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *,
_class"
 loc:@fully_connected_1/weights
°
2fully_connected_1/weights/Adam_1/Initializer/zerosFillBfully_connected_1/weights/Adam_1/Initializer/zeros/shape_as_tensor8fully_connected_1/weights/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*

index_type0*,
_class"
 loc:@fully_connected_1/weights
¬
 fully_connected_1/weights/Adam_1
VariableV2*
shared_name *,
_class"
 loc:@fully_connected_1/weights*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
З
'fully_connected_1/weights/Adam_1/AssignAssign fully_connected_1/weights/Adam_12fully_connected_1/weights/Adam_1/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@fully_connected_1/weights*
validate_shape(*
_output_shapes

:@@
™
%fully_connected_1/weights/Adam_1/readIdentity fully_connected_1/weights/Adam_1*
T0*,
_class"
 loc:@fully_connected_1/weights*
_output_shapes

:@@
©
/fully_connected_1/biases/Adam/Initializer/zerosConst*
valueB@*    *+
_class!
loc:@fully_connected_1/biases*
dtype0*
_output_shapes
:@
ґ
fully_connected_1/biases/Adam
VariableV2*
shared_name *+
_class!
loc:@fully_connected_1/biases*
	container *
shape:@*
dtype0*
_output_shapes
:@
щ
$fully_connected_1/biases/Adam/AssignAssignfully_connected_1/biases/Adam/fully_connected_1/biases/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@fully_connected_1/biases*
validate_shape(*
_output_shapes
:@
Я
"fully_connected_1/biases/Adam/readIdentityfully_connected_1/biases/Adam*
T0*+
_class!
loc:@fully_connected_1/biases*
_output_shapes
:@
Ђ
1fully_connected_1/biases/Adam_1/Initializer/zerosConst*
valueB@*    *+
_class!
loc:@fully_connected_1/biases*
dtype0*
_output_shapes
:@
Є
fully_connected_1/biases/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *+
_class!
loc:@fully_connected_1/biases*
	container 
€
&fully_connected_1/biases/Adam_1/AssignAssignfully_connected_1/biases/Adam_11fully_connected_1/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@fully_connected_1/biases
£
$fully_connected_1/biases/Adam_1/readIdentityfully_connected_1/biases/Adam_1*
_output_shapes
:@*
T0*+
_class!
loc:@fully_connected_1/biases
≥
0fully_connected_2/weights/Adam/Initializer/zerosConst*
valueB@*    *,
_class"
 loc:@fully_connected_2/weights*
dtype0*
_output_shapes

:@
ј
fully_connected_2/weights/Adam
VariableV2*
shared_name *,
_class"
 loc:@fully_connected_2/weights*
	container *
shape
:@*
dtype0*
_output_shapes

:@
Б
%fully_connected_2/weights/Adam/AssignAssignfully_connected_2/weights/Adam0fully_connected_2/weights/Adam/Initializer/zeros*
use_locking(*
T0*,
_class"
 loc:@fully_connected_2/weights*
validate_shape(*
_output_shapes

:@
¶
#fully_connected_2/weights/Adam/readIdentityfully_connected_2/weights/Adam*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes

:@
µ
2fully_connected_2/weights/Adam_1/Initializer/zerosConst*
valueB@*    *,
_class"
 loc:@fully_connected_2/weights*
dtype0*
_output_shapes

:@
¬
 fully_connected_2/weights/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *,
_class"
 loc:@fully_connected_2/weights*
	container *
shape
:@
З
'fully_connected_2/weights/Adam_1/AssignAssign fully_connected_2/weights/Adam_12fully_connected_2/weights/Adam_1/Initializer/zeros*
T0*,
_class"
 loc:@fully_connected_2/weights*
validate_shape(*
_output_shapes

:@*
use_locking(
™
%fully_connected_2/weights/Adam_1/readIdentity fully_connected_2/weights/Adam_1*
T0*,
_class"
 loc:@fully_connected_2/weights*
_output_shapes

:@
©
/fully_connected_2/biases/Adam/Initializer/zerosConst*
valueB*    *+
_class!
loc:@fully_connected_2/biases*
dtype0*
_output_shapes
:
ґ
fully_connected_2/biases/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@fully_connected_2/biases*
	container *
shape:
щ
$fully_connected_2/biases/Adam/AssignAssignfully_connected_2/biases/Adam/fully_connected_2/biases/Adam/Initializer/zeros*
T0*+
_class!
loc:@fully_connected_2/biases*
validate_shape(*
_output_shapes
:*
use_locking(
Я
"fully_connected_2/biases/Adam/readIdentityfully_connected_2/biases/Adam*
T0*+
_class!
loc:@fully_connected_2/biases*
_output_shapes
:
Ђ
1fully_connected_2/biases/Adam_1/Initializer/zerosConst*
valueB*    *+
_class!
loc:@fully_connected_2/biases*
dtype0*
_output_shapes
:
Є
fully_connected_2/biases/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *+
_class!
loc:@fully_connected_2/biases*
	container *
shape:
€
&fully_connected_2/biases/Adam_1/AssignAssignfully_connected_2/biases/Adam_11fully_connected_2/biases/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*+
_class!
loc:@fully_connected_2/biases
£
$fully_connected_2/biases/Adam_1/readIdentityfully_connected_2/biases/Adam_1*
T0*+
_class!
loc:@fully_connected_2/biases*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *oГ:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *wЊ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
valueB
 *wћ+2*
dtype0*
_output_shapes
: 
≠
-Adam/update_fully_connected/weights/ApplyAdam	ApplyAdamfully_connected/weightsfully_connected/weights/Adamfully_connected/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon@gradients/fully_connected/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0**
_class 
loc:@fully_connected/weights
•
,Adam/update_fully_connected/biases/ApplyAdam	ApplyAdamfully_connected/biasesfully_connected/biases/Adamfully_connected/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonAgradients/fully_connected/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*)
_class
loc:@fully_connected/biases
є
/Adam/update_fully_connected_1/weights/ApplyAdam	ApplyAdamfully_connected_1/weightsfully_connected_1/weights/Adam fully_connected_1/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/fully_connected_1/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@fully_connected_1/weights*
use_nesterov( *
_output_shapes

:@@*
use_locking( 
±
.Adam/update_fully_connected_1/biases/ApplyAdam	ApplyAdamfully_connected_1/biasesfully_connected_1/biases/Adamfully_connected_1/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/fully_connected_1/BiasAdd_grad/tuple/control_dependency_1*
T0*+
_class!
loc:@fully_connected_1/biases*
use_nesterov( *
_output_shapes
:@*
use_locking( 
є
/Adam/update_fully_connected_2/weights/ApplyAdam	ApplyAdamfully_connected_2/weightsfully_connected_2/weights/Adam fully_connected_2/weights/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonBgradients/fully_connected_2/MatMul_grad/tuple/control_dependency_1*
T0*,
_class"
 loc:@fully_connected_2/weights*
use_nesterov( *
_output_shapes

:@*
use_locking( 
±
.Adam/update_fully_connected_2/biases/ApplyAdam	ApplyAdamfully_connected_2/biasesfully_connected_2/biases/Adamfully_connected_2/biases/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/fully_connected_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@fully_connected_2/biases*
use_nesterov( *
_output_shapes
:
Ю
Adam/mulMulbeta1_power/read
Adam/beta1-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam/^Adam/update_fully_connected_1/biases/ApplyAdam0^Adam/update_fully_connected_1/weights/ApplyAdam/^Adam/update_fully_connected_2/biases/ApplyAdam0^Adam/update_fully_connected_2/weights/ApplyAdam*
T0*)
_class
loc:@fully_connected/biases*
_output_shapes
: 
°
Adam/AssignAssignbeta1_powerAdam/mul*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
: *
use_locking( 
†

Adam/mul_1Mulbeta2_power/read
Adam/beta2-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam/^Adam/update_fully_connected_1/biases/ApplyAdam0^Adam/update_fully_connected_1/weights/ApplyAdam/^Adam/update_fully_connected_2/biases/ApplyAdam0^Adam/update_fully_connected_2/weights/ApplyAdam*
T0*)
_class
loc:@fully_connected/biases*
_output_shapes
: 
•
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*)
_class
loc:@fully_connected/biases
ѕ
AdamNoOp^Adam/Assign^Adam/Assign_1-^Adam/update_fully_connected/biases/ApplyAdam.^Adam/update_fully_connected/weights/ApplyAdam/^Adam/update_fully_connected_1/biases/ApplyAdam0^Adam/update_fully_connected_1/weights/ApplyAdam/^Adam/update_fully_connected_2/biases/ApplyAdam0^Adam/update_fully_connected_2/weights/ApplyAdam
я
initNoOp^beta1_power/Assign^beta2_power/Assign#^fully_connected/biases/Adam/Assign%^fully_connected/biases/Adam_1/Assign^fully_connected/biases/Assign$^fully_connected/weights/Adam/Assign&^fully_connected/weights/Adam_1/Assign^fully_connected/weights/Assign%^fully_connected_1/biases/Adam/Assign'^fully_connected_1/biases/Adam_1/Assign ^fully_connected_1/biases/Assign&^fully_connected_1/weights/Adam/Assign(^fully_connected_1/weights/Adam_1/Assign!^fully_connected_1/weights/Assign%^fully_connected_2/biases/Adam/Assign'^fully_connected_2/biases/Adam_1/Assign ^fully_connected_2/biases/Assign&^fully_connected_2/weights/Adam/Assign(^fully_connected_2/weights/Adam_1/Assign!^fully_connected_2/weights/Assign
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
Д
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_02dc593f60684ab0ae247a6416b0b4c3/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
Х
save/SaveV2/tensor_namesConst*»
valueЊBїBbeta1_powerBbeta2_powerBfully_connected/biasesBfully_connected/biases/AdamBfully_connected/biases/Adam_1Bfully_connected/weightsBfully_connected/weights/AdamBfully_connected/weights/Adam_1Bfully_connected_1/biasesBfully_connected_1/biases/AdamBfully_connected_1/biases/Adam_1Bfully_connected_1/weightsBfully_connected_1/weights/AdamB fully_connected_1/weights/Adam_1Bfully_connected_2/biasesBfully_connected_2/biases/AdamBfully_connected_2/biases/Adam_1Bfully_connected_2/weightsBfully_connected_2/weights/AdamB fully_connected_2/weights/Adam_1*
dtype0*
_output_shapes
:
Л
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ї
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerfully_connected/biasesfully_connected/biases/Adamfully_connected/biases/Adam_1fully_connected/weightsfully_connected/weights/Adamfully_connected/weights/Adam_1fully_connected_1/biasesfully_connected_1/biases/Adamfully_connected_1/biases/Adam_1fully_connected_1/weightsfully_connected_1/weights/Adam fully_connected_1/weights/Adam_1fully_connected_2/biasesfully_connected_2/biases/Adamfully_connected_2/biases/Adam_1fully_connected_2/weightsfully_connected_2/weights/Adam fully_connected_2/weights/Adam_1*"
dtypes
2
С
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
Э
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*

axis *
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
Ш
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:*»
valueЊBїBbeta1_powerBbeta2_powerBfully_connected/biasesBfully_connected/biases/AdamBfully_connected/biases/Adam_1Bfully_connected/weightsBfully_connected/weights/AdamBfully_connected/weights/Adam_1Bfully_connected_1/biasesBfully_connected_1/biases/AdamBfully_connected_1/biases/Adam_1Bfully_connected_1/weightsBfully_connected_1/weights/AdamB fully_connected_1/weights/Adam_1Bfully_connected_2/biasesBfully_connected_2/biases/AdamBfully_connected_2/biases/Adam_1Bfully_connected_2/weightsBfully_connected_2/weights/AdamB fully_connected_2/weights/Adam_1
О
save/RestoreV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
п
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2
І
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@fully_connected/biases
Ђ
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*)
_class
loc:@fully_connected/biases
Ї
save/Assign_2Assignfully_connected/biasessave/RestoreV2:2*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
њ
save/Assign_3Assignfully_connected/biases/Adamsave/RestoreV2:3*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@fully_connected/biases
Ѕ
save/Assign_4Assignfully_connected/biases/Adam_1save/RestoreV2:4*
use_locking(*
T0*)
_class
loc:@fully_connected/biases*
validate_shape(*
_output_shapes
:@
ј
save/Assign_5Assignfully_connected/weightssave/RestoreV2:5*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@*
use_locking(
≈
save/Assign_6Assignfully_connected/weights/Adamsave/RestoreV2:6*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@
«
save/Assign_7Assignfully_connected/weights/Adam_1save/RestoreV2:7*
use_locking(*
T0**
_class 
loc:@fully_connected/weights*
validate_shape(*
_output_shapes

:@
Њ
save/Assign_8Assignfully_connected_1/biasessave/RestoreV2:8*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*+
_class!
loc:@fully_connected_1/biases
√
save/Assign_9Assignfully_connected_1/biases/Adamsave/RestoreV2:9*
T0*+
_class!
loc:@fully_connected_1/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
«
save/Assign_10Assignfully_connected_1/biases/Adam_1save/RestoreV2:10*
T0*+
_class!
loc:@fully_connected_1/biases*
validate_shape(*
_output_shapes
:@*
use_locking(
∆
save/Assign_11Assignfully_connected_1/weightssave/RestoreV2:11*
use_locking(*
T0*,
_class"
 loc:@fully_connected_1/weights*
validate_shape(*
_output_shapes

:@@
Ћ
save/Assign_12Assignfully_connected_1/weights/Adamsave/RestoreV2:12*
use_locking(*
T0*,
_class"
 loc:@fully_connected_1/weights*
validate_shape(*
_output_shapes

:@@
Ќ
save/Assign_13Assign fully_connected_1/weights/Adam_1save/RestoreV2:13*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*,
_class"
 loc:@fully_connected_1/weights
ј
save/Assign_14Assignfully_connected_2/biasessave/RestoreV2:14*
use_locking(*
T0*+
_class!
loc:@fully_connected_2/biases*
validate_shape(*
_output_shapes
:
≈
save/Assign_15Assignfully_connected_2/biases/Adamsave/RestoreV2:15*
use_locking(*
T0*+
_class!
loc:@fully_connected_2/biases*
validate_shape(*
_output_shapes
:
«
save/Assign_16Assignfully_connected_2/biases/Adam_1save/RestoreV2:16*
use_locking(*
T0*+
_class!
loc:@fully_connected_2/biases*
validate_shape(*
_output_shapes
:
∆
save/Assign_17Assignfully_connected_2/weightssave/RestoreV2:17*
T0*,
_class"
 loc:@fully_connected_2/weights*
validate_shape(*
_output_shapes

:@*
use_locking(
Ћ
save/Assign_18Assignfully_connected_2/weights/Adamsave/RestoreV2:18*
use_locking(*
T0*,
_class"
 loc:@fully_connected_2/weights*
validate_shape(*
_output_shapes

:@
Ќ
save/Assign_19Assign fully_connected_2/weights/Adam_1save/RestoreV2:19*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*,
_class"
 loc:@fully_connected_2/weights
в
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard "<
save/Const:0save/Identity:0save/restore_all (5 @F8"(
losses

mean_squared_error/value:0"
train_op

Adam"ј
regularization_losses¶
£
3fully_connected/kernel/Regularizer/l2_regularizer:0
5fully_connected_1/kernel/Regularizer/l2_regularizer:0
5fully_connected_2/kernel/Regularizer/l2_regularizer:0"Ф
model_variablesАэ
У
fully_connected/weights:0fully_connected/weights/Assignfully_connected/weights/read:024fully_connected/weights/Initializer/random_uniform:08
Ж
fully_connected/biases:0fully_connected/biases/Assignfully_connected/biases/read:02*fully_connected/biases/Initializer/zeros:08
Ы
fully_connected_1/weights:0 fully_connected_1/weights/Assign fully_connected_1/weights/read:026fully_connected_1/weights/Initializer/random_uniform:08
О
fully_connected_1/biases:0fully_connected_1/biases/Assignfully_connected_1/biases/read:02,fully_connected_1/biases/Initializer/zeros:08
Ы
fully_connected_2/weights:0 fully_connected_2/weights/Assign fully_connected_2/weights/read:026fully_connected_2/weights/Initializer/random_uniform:08
О
fully_connected_2/biases:0fully_connected_2/biases/Assignfully_connected_2/biases/read:02,fully_connected_2/biases/Initializer/zeros:08"Ш
trainable_variablesАэ
У
fully_connected/weights:0fully_connected/weights/Assignfully_connected/weights/read:024fully_connected/weights/Initializer/random_uniform:08
Ж
fully_connected/biases:0fully_connected/biases/Assignfully_connected/biases/read:02*fully_connected/biases/Initializer/zeros:08
Ы
fully_connected_1/weights:0 fully_connected_1/weights/Assign fully_connected_1/weights/read:026fully_connected_1/weights/Initializer/random_uniform:08
О
fully_connected_1/biases:0fully_connected_1/biases/Assignfully_connected_1/biases/read:02,fully_connected_1/biases/Initializer/zeros:08
Ы
fully_connected_2/weights:0 fully_connected_2/weights/Assign fully_connected_2/weights/read:026fully_connected_2/weights/Initializer/random_uniform:08
О
fully_connected_2/biases:0fully_connected_2/biases/Assignfully_connected_2/biases/read:02,fully_connected_2/biases/Initializer/zeros:08"Ж
	variablesшх
У
fully_connected/weights:0fully_connected/weights/Assignfully_connected/weights/read:024fully_connected/weights/Initializer/random_uniform:08
Ж
fully_connected/biases:0fully_connected/biases/Assignfully_connected/biases/read:02*fully_connected/biases/Initializer/zeros:08
Ы
fully_connected_1/weights:0 fully_connected_1/weights/Assign fully_connected_1/weights/read:026fully_connected_1/weights/Initializer/random_uniform:08
О
fully_connected_1/biases:0fully_connected_1/biases/Assignfully_connected_1/biases/read:02,fully_connected_1/biases/Initializer/zeros:08
Ы
fully_connected_2/weights:0 fully_connected_2/weights/Assign fully_connected_2/weights/read:026fully_connected_2/weights/Initializer/random_uniform:08
О
fully_connected_2/biases:0fully_connected_2/biases/Assignfully_connected_2/biases/read:02,fully_connected_2/biases/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
Ь
fully_connected/weights/Adam:0#fully_connected/weights/Adam/Assign#fully_connected/weights/Adam/read:020fully_connected/weights/Adam/Initializer/zeros:0
§
 fully_connected/weights/Adam_1:0%fully_connected/weights/Adam_1/Assign%fully_connected/weights/Adam_1/read:022fully_connected/weights/Adam_1/Initializer/zeros:0
Ш
fully_connected/biases/Adam:0"fully_connected/biases/Adam/Assign"fully_connected/biases/Adam/read:02/fully_connected/biases/Adam/Initializer/zeros:0
†
fully_connected/biases/Adam_1:0$fully_connected/biases/Adam_1/Assign$fully_connected/biases/Adam_1/read:021fully_connected/biases/Adam_1/Initializer/zeros:0
§
 fully_connected_1/weights/Adam:0%fully_connected_1/weights/Adam/Assign%fully_connected_1/weights/Adam/read:022fully_connected_1/weights/Adam/Initializer/zeros:0
ђ
"fully_connected_1/weights/Adam_1:0'fully_connected_1/weights/Adam_1/Assign'fully_connected_1/weights/Adam_1/read:024fully_connected_1/weights/Adam_1/Initializer/zeros:0
†
fully_connected_1/biases/Adam:0$fully_connected_1/biases/Adam/Assign$fully_connected_1/biases/Adam/read:021fully_connected_1/biases/Adam/Initializer/zeros:0
®
!fully_connected_1/biases/Adam_1:0&fully_connected_1/biases/Adam_1/Assign&fully_connected_1/biases/Adam_1/read:023fully_connected_1/biases/Adam_1/Initializer/zeros:0
§
 fully_connected_2/weights/Adam:0%fully_connected_2/weights/Adam/Assign%fully_connected_2/weights/Adam/read:022fully_connected_2/weights/Adam/Initializer/zeros:0
ђ
"fully_connected_2/weights/Adam_1:0'fully_connected_2/weights/Adam_1/Assign'fully_connected_2/weights/Adam_1/read:024fully_connected_2/weights/Adam_1/Initializer/zeros:0
†
fully_connected_2/biases/Adam:0$fully_connected_2/biases/Adam/Assign$fully_connected_2/biases/Adam/read:021fully_connected_2/biases/Adam/Initializer/zeros:0
®
!fully_connected_2/biases/Adam_1:0&fully_connected_2/biases/Adam_1/Assign&fully_connected_2/biases/Adam_1/read:023fully_connected_2/biases/Adam_1/Initializer/zeros:0*ѓ
serving_defaultЫ
?
input_placeholder*
input_placeholder:0€€€€€€€€€<
output2
fully_connected_2/BiasAdd:0€€€€€€€€€tensorflow/serving/predict