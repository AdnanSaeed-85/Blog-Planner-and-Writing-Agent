import streamlit as st
from main import app
from pathlib import Path
import time

# Set page config
st.set_page_config(page_title="Blog Generator", layout="wide")

# Function to get all markdown files
def get_all_blogs():
    """Get list of all .md files in current directory"""
    md_files = sorted(Path('.').glob('*.md'), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in md_files]

# Function to read blog content
def read_blog(filename):
    """Read content of a markdown file"""
    return Path(filename).read_text(encoding='utf-8')

# Sidebar
st.sidebar.markdown("## 📝 Generate New Blog")
st.sidebar.divider()

input_text = st.sidebar.text_input('Enter Topic', key='topic_input')

if st.sidebar.button('Generate Blog', type='primary'):
    if input_text:
        st.session_state.generating = True
        st.session_state.plan_data = None
        st.session_state.research_data = None
        st.rerun()
    else:
        st.sidebar.error('Please enter a topic!')

st.sidebar.divider()
st.sidebar.markdown("## 📚 Your Blogs")

# Get all blogs
all_blogs = get_all_blogs()

if all_blogs:
    if 'selected_blog' not in st.session_state:
        st.session_state.selected_blog = all_blogs[0]
    
    selected = st.sidebar.radio(
        "Select a blog to view:",
        all_blogs,
        index=all_blogs.index(st.session_state.selected_blog) if st.session_state.selected_blog in all_blogs else 0,
        label_visibility='collapsed'
    )
    
    st.session_state.selected_blog = selected
    
    if st.sidebar.button('🗑️ Delete Selected Blog', type='secondary'):
        Path(selected).unlink()
        st.success(f'Deleted: {selected}')
        st.rerun()
else:
    st.sidebar.info('No blogs yet. Generate one!')

# Main content area
st.markdown('# 🚀 Blogging Deep Agent')
st.divider()

# Check if we're generating a new blog
if 'generating' in st.session_state and st.session_state.generating:
    st.markdown(f"## Generating blog about: **{input_text}**")
    st.divider()
    
    # Task bar with expanders
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📋 Planning", expanded=False):
            plan_placeholder = st.empty()
            if st.session_state.plan_data:
                plan_placeholder.json(st.session_state.plan_data)
            else:
                plan_placeholder.info("Waiting for plan...")
    
    with col2:
        with st.expander("🔍 Research", expanded=False):
            research_placeholder = st.empty()
            if st.session_state.research_data:
                if st.session_state.research_data == "no_research":
                    research_placeholder.info("No online research was performed for this topic.")
                else:
                    research_placeholder.json(st.session_state.research_data)
            else:
                research_placeholder.info("Waiting for research phase...")
    
    st.divider()
    
    # Fixed status at top
    status_placeholder = st.empty()
    status_placeholder.info('✍️ Writing sections...')
    
    # Create placeholder for streaming content
    stream_placeholder = st.empty()
    
    accumulated_content = ""
    
    try:
        for event in app.stream({'topic': input_text, 'sections': []}):
            
            # Capture orchestrator plan
            if 'orchestrator' in event:
                plan_dict = event['orchestrator'].get('plan')
                if plan_dict:
                    st.session_state.plan_data = plan_dict
                    plan_placeholder.json(plan_dict)
            
            # Capture research results
            if 'research' in event:
                evidence = event['research'].get('evidence', [])
                if evidence:
                    st.session_state.research_data = evidence
                    research_placeholder.json(evidence)
                else:
                    st.session_state.research_data = "no_research"
                    research_placeholder.info("No online research was performed for this topic.")
            
            # Stream worker sections
            if 'worker' in event:
                sections_data = event['worker'].get('sections', [])
                if sections_data:
                    for task_id, section_content in sections_data:
                        accumulated_content += section_content + "\n\n"
                        stream_placeholder.markdown(accumulated_content)
        
        status_placeholder.success('✅ Blog generated successfully!')
        time.sleep(1)
        
        st.session_state.generating = False
        st.rerun()
        
    except Exception as e:
        st.error(f'Error generating blog: {str(e)}')
        st.session_state.generating = False

elif all_blogs and st.session_state.selected_blog:
    blog_content = read_blog(st.session_state.selected_blog)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f'📄 Currently viewing: **{st.session_state.selected_blog}**')
    with col2:
        st.download_button(
            label='⬇️ Download',
            data=blog_content,
            file_name=st.session_state.selected_blog,
            mime='text/markdown'
        )
    
    st.divider()
    st.markdown(blog_content)
else:
    st.info("👈 Enter a topic in the sidebar to generate your first blog!")