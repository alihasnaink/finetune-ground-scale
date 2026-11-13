# Complementary-pair examples

(fixed before looking at the examples): for each run, suite and
question type, pairs are taken in pair-ID order and the first pair of each category is shown:
(a) both answers correct with the correct images, (b) identical answers for the two images of the
pair (image-insensitive), (c) different answers for the two images but at least one of them wrong
(image-sensitive error). A pair is shown at most once per run; categories without a matching pair
are skipped. (The first version's category (c), "the swap makes the model give the partner's answer",
was replaced after the first full run because it is implied by (a) and selected the same pairs.)

## vlm196

### grounding / counting / both correct: `gtest_coun_00001`
**Q:** How many trains can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/545040 | 2 | `2` | `4` | `1` | `The number of trains you can see depends on the location, time of day,` |
| val2014/236837 | 4 | `4` | `2` | `1` | `The number of trains you can see depends on the location, time of day,` |

### grounding / counting / image-insensitive: `gtest_coun_00002`
**Q:** How many microwaves can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/103870 | 1 | `1` | `1` | `1` | `You cannot see microwaves. Microwaves are a type of electromagnetic radiation` |
| val2014/8170 | 2 | `1` | `1` | `0` | `You cannot see microwaves. Microwaves are a type of electromagnetic radiation` |

### grounding / counting / image-sensitive error: `gtest_coun_00000`
**Q:** How many kites can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/400033 | 2 | `2` | `4` | `1` | `There are **no kites** visible in the sky.` |
| val2014/142552 | 3 | `4` | `2` | `1` | `There are **no kites** visible in the sky.` |

### grounding / presence / both correct: `gtest_pres_00001`
**Q:** Does this image contain a knife?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/476029 | yes | `yes` | `no` | `no` | `No, the image does not contain a knife.` |
| val2014/182523 | no | `no` | `yes` | `no` | `No, the image does not contain a knife.` |

### grounding / presence / image-insensitive: `gtest_pres_00000`
**Q:** Does this image contain a backpack?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/171548 | yes | `no` | `no` | `no` | `Yes, the image contains a backpack.` |
| val2014/205647 | no | `no` | `no` | `no` | `Yes, the image contains a backpack.` |

### vqa / number / both correct: `vqa_19404000_140636013`
**Q:** How many birds are there?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/19404 | 1 | `1` | `2` | `0` | `There are no birds in the sky.` |
| val2014/140636 | 2 | `2` | `1` | `0` | `There are no birds in the sky.` |

### vqa / number / image-insensitive: `vqa_110972001_194352007`
**Q:** How many legs is the bear standing on?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/110972 | 4 | `2` | `2` | `4` | `The question is ambiguous and unclear. However, if we assume that the bear is` |
| val2014/194352 | 2 | `2` | `2` | `2` | `The question is ambiguous and unclear. However, if we assume that the bear is` |

### vqa / number / image-sensitive error: `vqa_156184002_331455005`
**Q:** How many boats are in the water?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/156184 | 0 | `1` | `2` | `0` | `The question is a bit ambiguous, but if we assume it's asking about the` |
| val2014/331455 | 2 | `2` | `1` | `0` | `The question is a bit ambiguous, but if we assume it's asking about the` |

### vqa / other / both correct: `vqa_12326004_214204010`
**Q:** Which room is this?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/12326 | kitchen | `kitchen` | `bathroom` | `lunch` | `The room is **Kitchen**.` |
| val2014/214204 | bathroom | `bathroom` | `kitchen` | `dining` | `The room is **Kitchen**.` |

### vqa / other / image-insensitive: `vqa_109937000_53952008`
**Q:** What color is the girl's shirt?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/109937 | white | `white` | `white` | `white` | `The color of the girl's shirt is not specified in the question. If you` |
| val2014/53952 | pink | `white` | `white` | `white` | `The color of the girl's shirt is not specified in the question. If you` |

### vqa / other / image-sensitive error: `vqa_188945000_214244003`
**Q:** What is this person riding?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/188945 | skateboard | `skateboard` | `ski` | `elephant` | `The question is ambiguous and unclear. If you're referring to a person riding something` |
| val2014/214244 | skis | `ski` | `skateboard` | `cow` | `The question is ambiguous and unclear. If you're referring to a person riding something` |

### vqa / yes/no / both correct: `vqa_159791000_113082001`
**Q:** Does the owner of the fridge like stickers?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/159791 | yes | `yes` | `no` | `no` | `No, the owner of the fridge does not like stickers.` |
| val2014/113082 | no | `no` | `yes` | `no` | `No, the owner of the fridge does not like stickers.` |

### vqa / yes/no / image-insensitive: `vqa_113235001_195862007`
**Q:** Is this a horse?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/113235 | no | `yes` | `yes` | `no` | `No, this is not a horse.` |
| val2014/195862 | yes | `yes` | `yes` | `no` | `No, this is not a horse.` |

### vqa / yes/no / image-sensitive error: `vqa_161505000_89880045`
**Q:** Is this the same dog?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/161505 | yes | `no` | `yes` | `no` | `No, this is not the same dog.` |
| val2014/89880 | no | `yes` | `no` | `no` | `No, this is not the same dog.` |

## vlm49

### grounding / counting / both correct: `gtest_coun_00000`
**Q:** How many kites can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/400033 | 2 | `2` | `3` | `1` | `1` |
| val2014/142552 | 3 | `3` | `2` | `1` | `1` |

### grounding / counting / image-insensitive: `gtest_coun_00002`
**Q:** How many microwaves can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/103870 | 1 | `1` | `1` | `0` | `0` |
| val2014/8170 | 2 | `1` | `1` | `0` | `0` |

### grounding / counting / image-sensitive error: `gtest_coun_00001`
**Q:** How many trains can you see?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/545040 | 2 | `1` | `4` | `3` | `1` |
| val2014/236837 | 4 | `4` | `1` | `3` | `1` |

### grounding / presence / both correct: `gtest_pres_00001`
**Q:** Does this image contain a knife?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/476029 | yes | `yes` | `no` | `no` | `yes` |
| val2014/182523 | no | `no` | `yes` | `no` | `yes` |

### grounding / presence / image-insensitive: `gtest_pres_00000`
**Q:** Does this image contain a backpack?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/171548 | yes | `no` | `no` | `no` | `yes` |
| val2014/205647 | no | `no` | `no` | `no` | `yes` |

### vqa / number / both correct: `vqa_19404000_140636013`
**Q:** How many birds are there?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/19404 | 1 | `1` | `2` | `0` | `1` |
| val2014/140636 | 2 | `2` | `1` | `1` | `1` |

### vqa / number / image-insensitive: `vqa_110972001_194352007`
**Q:** How many legs is the bear standing on?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/110972 | 4 | `4` | `4` | `4` | `2` |
| val2014/194352 | 2 | `4` | `4` | `2` | `2` |

### vqa / number / image-sensitive error: `vqa_218886001_164969007`
**Q:** How many pizzas are in the picture?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/218886 | 2 | `1` | `2` | `0` | `1` |
| val2014/164969 | 1 | `2` | `1` | `0` | `1` |

### vqa / other / both correct: `vqa_112568002_116046003`
**Q:** Is this indoors or outdoors?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/112568 | outdoors | `outdoors` | `indoors` | `outdoors` | `indoors` |
| val2014/116046 | indoors | `indoors` | `outdoors` | `outdoors` | `indoors` |

### vqa / other / image-insensitive: `vqa_109937000_53952008`
**Q:** What color is the girl's shirt?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/109937 | white | `white` | `white` | `white` | `red` |
| val2014/53952 | pink | `white` | `white` | `white` | `red` |

### vqa / other / image-sensitive error: `vqa_156246001_380088003`
**Q:** What is beside the boy?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/156246 | elephant | `elephant` | `calf` | `book` | `the girl` |
| val2014/380088 | cow | `calf` | `elephant` | `chair` | `the girl` |

### vqa / yes/no / both correct: `vqa_113235001_195862007`
**Q:** Is this a horse?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/113235 | no | `no` | `yes` | `no` | `no` |
| val2014/195862 | yes | `yes` | `no` | `no` | `no` |

### vqa / yes/no / image-insensitive: `vqa_159791000_113082001`
**Q:** Does the owner of the fridge like stickers?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/159791 | yes | `no` | `no` | `no` | `no` |
| val2014/113082 | no | `no` | `no` | `no` | `no` |

### vqa / yes/no / image-sensitive error: `vqa_355871000_164330003`
**Q:** Do the people look happy?

| image | references | correct | complementary | mismatched | no image |
|---|---|---|---|---|---|
| val2014/355871 | yes | `no` | `yes` | `no` | `no` |
| val2014/164330 | no | `yes` | `no` | `no` | `no` |

