# IMPLEMENTATION_PLAN.md
## 目的
現在の「リアルタイム音声フォルマント検出 + GUI表示」Rust(cargo)プロジェクトに対し、

- **Kalmanフィルタ**による時系列追跡（平滑化 + 欠損耐性）
- **Viterbiアルゴリズム**による離散状態系列の最尤推定（子音混入で崩れにくい遷移制約）

の **2つの解決策**を実装し、**UIで切り替え可能**にする。

このファイルは **Codex にそのまま渡すための変更指示計画書**（実装ロードマップ + 仕様 + 追加ファイル設計 + 受け入れ基準）。


---

## 現状認識（前提）
- 長い母音（単音）では安定するが、文章中の**子音**（無声化・破裂・摩擦）で
  - F1/F2が瞬間的に飛ぶ
  - 推定が欠損したり誤検出ピークを拾う
  - フレーム間の連続性が崩れてGUIが荒れる
- 現在の推定器は「各フレーム独立」に近く、**時間的整合性**をモデル化していない可能性が高い。

ここに「時間モデル」を足すのが今回のゴール。


---

## 追加する機能（ユーザー向け仕様）
### UI仕様
- 推定モードをUIで切り替え：
  - `Raw`（現状の生推定）
  - `Kalman`（連続追跡）
  - `Viterbi`（離散最尤系列）
- 変更は即時反映（再起動不要）。
- モードごとに簡単なパラメータ調整UI（最小限でOK）：
  - Kalman: `process_noise`, `measurement_noise`, `max_jump_hz`
  - Viterbi: `state_grid_step_hz`, `transition_sigma_hz`, `dropout_cost`
- 表示：
  - 現在値（F1/F2/F3）
  - Raw vs Filtered を重ね描画できるとデバッグが早い（可能なら）。


---

## 設計方針（実装の骨格）
「推定（measurement）」と「追跡（tracking）」を分離する。

- **measurement**: 既存の各フレーム推定（LPCピーク/スペクトルピーク等）から  
  `Option<Formants>` を出す（欠損は None）
- **tracking**: measurement を受け取り、時間整合性を使って  
  `Formants`（または `Option<Formants>`）を出す

この分離により、Raw/Kalman/Viterbi を同じ差し替えポイントで切替できる。


---

## データ構造（追加/整理）
### Formants 型
- 既存にあるなら流用。なければ追加。

```rust
#[derive(Clone, Copy, Debug, Default)]
pub struct Formants {
    pub f1_hz: f32,
    pub f2_hz: f32,
    pub f3_hz: f32, // 使わないなら optional でも良いが、将来のため固定で持つのが楽
}

