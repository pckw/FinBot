def fill_prompt_template(template: dict, content: dict):
            result = template
            for key, value in content.items():
                result['content'] = template['content'].replace("{" + key + "}", value)
            return result