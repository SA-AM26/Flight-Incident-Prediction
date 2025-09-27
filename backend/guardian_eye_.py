streamlit.errors.StreamlitDuplicateElementKey: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

Traceback:
File "/mount/src/flight-incident-prediction/guardian_eye_.py", line 1242, in <module>
    main()
    ~~~~^^
File "/mount/src/flight-incident-prediction/guardian_eye_.py", line 1239, in main
    create_guardian_eye_streamlit()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
File "/mount/src/flight-incident-prediction/guardian_eye_.py", line 938, in create_guardian_eye_streamlit
    st.plotly_chart(fig, use_container_width=True, key="3d_globe")
    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/runtime/metrics_util.py", line 443, in wrapped_func
    result = non_optional_func(*args, **kwargs)
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/plotly_chart.py", line 565, in plotly_chart
    plotly_chart_proto.id = compute_and_register_element_id(
                            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        "plotly_chart",
        ^^^^^^^^^^^^^^^
    ...<8 lines>...
        use_container_width=use_container_width,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/utils.py", line 265, in compute_and_register_element_id
    _register_element_id(ctx, element_type, element_id)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/adminuser/venv/lib/python3.13/site-packages/streamlit/elements/lib/utils.py", line 145, in _register_element_id
    raise StreamlitDuplicateElementKey(user_key)
