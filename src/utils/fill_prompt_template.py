def fill_prompt_template(template: dict, content: dict) -> dict:
            """
            Replaces placeholders in a template dictionary with corresponding values from a content dictionary.

            Args:
                template (dict): The dictionary template containing placeholders.
                content (dict): The dictionary containing values to replace the placeholders.

            Returns:
                dict: The resulting dictionary with replaced values.
            """
            result = template
            for key, value in content.items():
                result['content'] = template['content'].replace("{" + key + "}", value)
            return result