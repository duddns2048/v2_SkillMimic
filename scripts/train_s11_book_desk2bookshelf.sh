CUDA_VISIBLE_DEVICES=1 python skillmimic/run.py \
  --experiment Move_book_from_desk_to_bookshelf\
  --task  SkillMimicParahomeLocalHistRISBuffernode\
  --episode_length 60 \
  --cfg_env skillmimic/data/cfg/parahome_sm/parahome_s11_book_desk2table_hist60_noisyinit_simpara.yaml \
  --cfg_train skillmimic/data/cfg/train/rlg/parahome.yaml \
  --motion_file skillmimic/data/motions/ParaHome/Move_book_from_desk_to_bookshelf/ \
  --reweight --reweight_alpha 1.0 \
  --state_init_random_prob 0.1 \
  --state_switch_prob 0.0 \
  --hist_length 60 \
  --history_embedding_size 3 \
  --hist_ckpt hist_encoder/ParaHome/hist_model.ckpt \
  --obj_trans /home/youngwoon/github/v2_SkillMimic/seq/s11/object_transformations.pkl \
  --headless
  # --resume_from
  # --wandb_id


    # parahome_place_kettle_hist60_noisyinit_simpara.yaml 
    #   in_scene_obj_static
    #   in_scene_obj_dynamic
    # --experiment
    # --obj_trans
    # --motion_file skillmimic/data/motions/Drink_from_cup/ \
    # --motion_file skillmimic/data/motions/Move_book_from_desk_to_bookshelf/ \
    # --motion_file skillmimic/data/motions/Move_chair_from_desk_to_table/ \
    # --motion_file skillmimic/data/motions/Move_cup_from_desk_to_table/ \
    # --motion_file skillmimic/data/motions/Move_cup_from_table_to_desk/ \
    # --motion_file skillmimic/data/motions/Move_kettle_from_table_to_desk/ \
    # --motion_file skillmimic/data/motions/Move_pan_from_gas_stove_to_table/ \