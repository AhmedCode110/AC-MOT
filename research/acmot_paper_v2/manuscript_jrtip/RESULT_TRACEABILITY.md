# سجل تتبّع نتائج مخطوط (JRTIP)

هذا السجل يربط النسخة الرئيسية المضغوطة بملفات السجل المجمدة. لا يشغّل تجارب علمية، ولا يعيد اشتقاق النتائج من مصدر جديد.

## هوية التجميد

- الوسم العلمي: `acmot-v3-exp-2026-10-04-freeze`.
- بيان التجميد الحديث: `research/acmot_paper_v2/controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json`.
- الالتزام المسجل لوحدة التحكم المجمدة: `2ed0697325db3ce3f1680889baaef3c77d8f573e`.
- قيمة (SHA256) للكاشف: `c6f98247e7c731a567aaf37ca7b808beff588b687623e028b25a2e1d846e96e7` من `FREEZE_MANIFEST_V3_ADAPTIVE.json -> detector.sha256`.
- اصطلاح العرض: القيم القديمة المطبّعة تضرب في 100 عند العرض كنسب مئوية؛ التجميعات الحديثة مخزنة أصلاً بصيغة العرض المئوية؛ الجداول الرئيسية تقرّب إلى ثلاثة أرقام عشرية ما لم يُذكر غير ذلك.

## سجل عناصر النسخة الرئيسية

| العنصر | المصدر المباشر والمفتاح | المعالجة |
|---|---|---|
| جدول 1: الملخص التاريخي المضغوط | `research/paper_split/evidence/legacy/OLD_ACMOT_COMPONENT_ABLATION.csv`; و`FINAL_TEST_RESULTS_3WORKER.json`; و`UAVDT_FINAL_COMPARISON.json` | صفوف مختارة للملخص التاريخي؛ الصفوف الكاملة في (S1) |
| جدول 2: خريطة أفعال (v3) | `research/acmot_paper_v2/controller_search/FREEZE_MANIFEST_V3_ADAPTIVE.json -> controller.action_mapping` | نقل مباشر للثلاثيات المجمدة |
| جدول 3: تجميع (VisDrone) الحديث | `BASELINE_SYSTEMS_RESULT.json -> YOLO11m+ByteTrack.aggregate`, `YOLO11m+OATrack.aggregate`; و`SECOND_VAL_READ_V3_RESULT.json -> AC-MOT-v3+YOLO11m+ByteTrack.aggregate`, `AC-MOT-v3+YOLO11m+OATrack.aggregate` | قيم تجميع مباشرة، مع تقريب العرض |
| جدول 4: (bootstrap) الحديث المختصر | `BOOTSTRAP_V3_RESULT.json -> bytetrack_v3_vs_system1.<metric>` و`oatrack_v3_vs_system2.<metric>` | حقول `diff`, `ci_lo`, `ci_hi` مباشرة؛ عدد السحوبات 5000 وبذرة 42 |
| الشكل 1: الدافع | `manuscript_jrtip/figures/fig_concept.tex` | رسم (TikZ) مبني على مفهوم نقطة التشغيل |
| الشكل 2: البنية | `manuscript_jrtip/figures/fig_architecture.tex` | تدفق (AC--MOT) الأصلي وفصل الكاشف عن المتعقّب |
| الشكل 3: آلة الحالات | `manuscript_jrtip/figures/fig_state_machine.tex` | الحالات والحواجز السببية لـ (v3) |
| الشكل 4: النتائج الحديثة | `manuscript_jrtip/figures/fig_modern_results.tex` | القيم الحديثة نفسها في جدول 3 |
| الخوارزمية 1: وحدة تحكم (v3) | `manuscript_jrtip/draft_sections/05_modern_acmot_v3.tex`, الخوارزمية `alg:v3` | الخوارزمية المجمدة في المتن |
| الخوارزمية الأصلية | `supplementary.tex`, القسم (S3.1) | نُقلت إلى المعلومات التكميلية؛ بقيت المعادلات وترتيب المعلومات في المتن |

لا يحتوي النص الرئيسي الحالي على شكل منفصل لزمن التشغيل أو للنقل إلى (UAVDT). القيم الرقمية لهذين الجزأين تبقى في النثر وتتتبّع إلى السجلات التالية.

## مفاتيح النتائج والضوابط

| الادعاء أو الفقرة | المصدر المباشر |
|---|---|
| نتائج الإسناد المطابق | `FINAL_ACMOT_SYSTEMS_RESULT.json -> AC-MOT+YOLO11m+<host>.aggregate` |
| ضابط التوقيت المخلوط | `SHUFFLED_CONTROL_RESULT.json -> causal_cv_mean`, `shuffled_cv_mean`, `seed` |
| زمن التشغيل الأساسي | `RUNTIME_BENCHMARK_RESULT.json -> results[]` |
| زمن تشغيل (v3) | `V3_RUNTIME_BENCHMARK_RESULT.json -> results[]` |
| النقل الحديث إلى (UAVDT) | `UAVDT_TRANSFER_RESULT.json -> <system>.aggregate` |
| اختيار الإشارات والعتبات وخريطة الأفعال | `FREEZE_MANIFEST_V3_ADAPTIVE.json` وسجل التجميد المرتبط |
| تفاصيل التحقق الثاني، الاختيار، والبروتوكول | `SECOND_VAL_READ_V3_RESULT.json`, `PAPER_READY_RESULTS.md`, و`EVIDENCE_MAP.md` |

## المراسي العددية الظاهرة في المتن

### التجميع الحديث

| النظام | HOTA | MOTA | IDF1 | IDS | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| خط أساس (ByteTrack) | 41.586244 | 9.876096 | 48.569072 | 1341 | 39274 | 24121 |
| (v3) مع (ByteTrack) | 44.523254 | 27.871363 | 53.929197 | 1016 | 24892 | 25902 |
| خط أساس (OATrack) | 46.769659 | 28.165112 | 57.106850 | 773 | 25394 | 25432 |
| (v3) مع (OATrack) | 48.095154 | 34.482807 | 59.364442 | 604 | 21400 | 25057 |

تُعرض هذه القيم في جدول 3 إلى ثلاثة أرقام عشرية. فواصل (bootstrap) الرئيسية في جدول 4 هي: (ByteTrack) فرق (HOTA) قدره `+2.937` وفاصل `[+2.356,+3.375]`، وفرق (MOTA) `+17.995` وفاصل `[+13.541,+23.282]`؛ و(OATrack) فرق (HOTA) `+1.325` وفاصل `[+0.200,+2.264]`، وفرق (MOTA) `+6.318` وفاصل `[+4.741,+7.879]`. تظهر بقية المقاييس الرئيسية في الجدول نفسه، بينما تفاصيل (FP/FN) الكاملة في (S2.1).

### الإسناد

- الضابط الساكن المطابق: (HOTA) `44.530` لـ (ByteTrack) و`48.269` لـ (OATrack)؛ وقيم (v3) المقابلة `44.523` و`48.095`.
- ضابط التوقيت المخلوط: متوسط التطوير `2.446957` للجدول السببي و`2.617228` للجدول المخلوط، مع `seed=44` و`causal_beats_shuffled=false`.
- هذه الأرقام لا تثبت أن التوقيت العشوائي أفضل على نحو عام، ولا تعزل مكسباً سببياً إضافياً لتوقيت التبديل الدقيق.

### الزمن والنقل

- قياس (v3) على (Tesla T4) يتتبع مباشرة إلى `V3_RUNTIME_BENCHMARK_RESULT.json`; وتبقى القيم الرئيسية في `draft_sections/08_runtime.tex`: متوسطا `13.45` و`15.27` إطاراً/ثانية، وزمنا تحكم قدرهما `4.3` و`4.2` مللي ثانية، مع قيد اختلاف أعمال قراءة الصورة بين أداتي القياس.
- النقل إلى (UAVDT) يتتبع إلى `UAVDT_TRANSFER_RESULT.json`: (ByteTrack) خط أساس/‏(v3) (HOTA/MOTA/IDF1/IDS) = `45.923/-1.573/58.400/601` ثم `47.004/16.028/61.767/486`؛ و(OATrack) = `47.728/19.607/62.101/517` ثم `47.640/22.873/62.931/486`.
- النقل عبارة عن 20 تسلسلاً و16592 إطاراً و49776 مروراً أمام الكاشف، تشغيل واحد لكل متعقّب، دون إعادة ضبط أو (bootstrap).

## ملاحظة التدقيق الخاصة بالبروتوكول

السجل `BASELINE_SYSTEMS_RESULT.json -> meta.protocol` يذكر `official (tools/v6/eval_official.py)`، بينما يصف `PAPER_READY_RESULTS.md` وسرد التجميد والمخطوط المقارنة الحديثة بأنها تستخدم بروتوكول المشروع المصنف على أنه (class-agnostic). هذا التعارض في بيانات المصدر محفوظ كمسألة مراجعة للمؤلفين؛ لا يدّعي المخطوط قابلية المقارنة مع لوحة (VisDrone) الرسمية. لم تُجرَ أي تجربة جديدة أثناء إعداد النسخة المضغوطة.
