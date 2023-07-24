<?php
	$sample = "Hello World!";
	header("Strict-Transport-Security: max-age=31536000; includeSubDomains");
?>

<!doctype html>

<html lang="en">
<head>
  <meta charset="utf-8">
  <title>III DevOps Sample</title>
  <style>
      h3{text-align: center;}
  </style>
</head>
<body>
  <h3><? echo $sample ?></h3>
</body>
</html>